import os
import pickle
import faiss
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class HybridRetriever:
    def __init__(self, 
                 vector_dir: str = "data/indices/vector", 
                 keyword_dir: str = "data/indices/keyword",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 rrf_k: int = 60):
        """
        Initializes the HybridRetriever by loading indices and models.
        """
        print("Initializing HybridRetriever...")
        self.rrf_k = rrf_k
        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        # Resource checking
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('tokenizers/punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

        # Data structures for indices
        self.stores = {} # {religion_name: {"vector_index": ..., "vector_docs": ..., "bm25": ..., "keyword_docs": ...}}
        self._load_indices(vector_dir, keyword_dir)

    def _load_indices(self, vector_dir: str, keyword_dir: str):
        """
        Loads all indices and document stores into memory.
        """
        if not os.path.exists(vector_dir) or not os.listdir(vector_dir):
            raise FileNotFoundError(f"Vector index directory {vector_dir} is empty or missing.")
        
        # We assume standard naming: {name}.faiss, {name}_bm25.pkl, {name}_docs.pkl
        # First, identify distinct sources (e.g., 'bible', 'bhagavad_gita')
        sources = set(f.replace(".faiss", "") for f in os.listdir(vector_dir) if f.endswith(".faiss"))
        
        for name in sources:
            print(f"Loading indices for: {name}...")
            source_data = {}
            
            # 1. Vector Index
            vec_path = os.path.join(vector_dir, f"{name}.faiss")
            vec_docs_path = os.path.join(vector_dir, f"{name}_docs.pkl")
            if os.path.exists(vec_path) and os.path.exists(vec_docs_path):
                source_data["vector_index"] = faiss.read_index(vec_path)
                with open(vec_docs_path, "rb") as f:
                    source_data["vector_docs"] = pickle.load(f)
            
            # 2. Keyword Index
            kw_path = os.path.join(keyword_dir, f"{name}_bm25.pkl")
            kw_docs_path = os.path.join(keyword_dir, f"{name}_docs.pkl")
            if os.path.exists(kw_path) and os.path.exists(kw_docs_path):
                with open(kw_path, "rb") as f:
                    source_data["bm25"] = pickle.load(f)
                with open(kw_docs_path, "rb") as f:
                    source_data["keyword_docs"] = pickle.load(f)
            
            if source_data:
                self.stores[name] = source_data
            else:
                print(f"Warning: Could not load complete index set for {name}")

    def _preprocess(self, text: str) -> List[str]:
        """Simple tokenizer for BM25 queries."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = word_tokenize(text)
        return [t for t in tokens if t not in self.stop_words]

    def _semantic_search(self, query_vector: np.ndarray, source_name: str, top_k: int) -> List[Tuple[int, float]]:
        """Internal semantic search."""
        with tracer.start_as_current_span("semantic_search") as span:
            span.set_attribute("source", source_name)
            span.set_attribute("top_k", top_k)
            index = self.stores[source_name].get("vector_index")
            if index is None: return []
            distances, indices = index.search(query_vector, top_k)
            results = list(zip(indices[0], distances[0]))
            span.set_attribute("result_count", len(results))
            return results

    def _keyword_search(self, tokenized_query: List[str], source_name: str, top_k: int) -> List[Tuple[int, float]]:
        """Internal keyword search."""
        with tracer.start_as_current_span("keyword_search") as span:
            span.set_attribute("source", source_name)
            span.set_attribute("top_k", top_k)
            bm25 = self.stores[source_name].get("bm25")
            if bm25 is None: return []
            scores = bm25.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:top_k]
            results = [(idx, scores[idx]) for idx in top_n if scores[idx] > 0]
            span.set_attribute("result_count", len(results))
            return results

    def _format_citation(self, metadata: Dict[str, Any], source_name: str) -> str:
        """
        Generates a consistent citation string based on source metadata.
        """
        if source_name == "bible":
            book = metadata.get("book", "Unknown")
            chapter = metadata.get("chapter", "?")
            verse = metadata.get("verse", "?")
            return f"{book} {chapter}:{verse}"
        elif source_name == "bhagavad_gita":
            chapter = metadata.get("chapter", "?")
            verse = metadata.get("verse", "?")
            return f"Bhagavad Gita {chapter}.{verse}"
        else:
            # Fallback for generic sources
            book = metadata.get("book", source_name.capitalize())
            chapter = metadata.get("chapter", "")
            verse = metadata.get("verse", "")
            citation = book
            if chapter: citation += f" {chapter}"
            if verse: citation += f":{verse}"
            return citation.strip()

    def get_top_k(self, query: str, religion: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top K results using Hybrid Search (Semantic + Keyword) fused with RRF,
        followed by a second-stage Cross-Encoder re-ranking.
        """
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("query", query)
            span.set_attribute("religion_filter", religion or "all")
            span.set_attribute("top_k", top_k)

            # Determine target sources
            target_sources = [religion] if religion and religion in self.stores else list(self.stores.keys())

            if not target_sources:
                print(f"No valid sources found for filter: {religion}")
                return []

            # Generate inputs
            query_vector = self.model.encode([query]).astype("float32")
            tokenized_query = self._preprocess(query)

            # 1. Candidate Generation (Stage 1)
            # Fetch more candidates for re-ranking (e.g., top_k * 4, min 20)
            rerank_candidate_n = max(top_k * 4, 20)
            
            # Run searches in parallel for efficiency
            raw_results = []
            with ThreadPoolExecutor() as executor:
                semantic_futures = {executor.submit(self._semantic_search, query_vector, s, rerank_candidate_n): (s, "semantic") for s in target_sources}
                keyword_futures = {executor.submit(self._keyword_search, tokenized_query, s, rerank_candidate_n): (s, "keyword") for s in target_sources}

                for future in semantic_futures:
                    source, method = semantic_futures[future]
                    raw_results.append((source, method, future.result()))
                for future in keyword_futures:
                    source, method = keyword_futures[future]
                    raw_results.append((source, method, future.result()))

            # Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            for source, method, results in raw_results:
                for rank, (doc_idx, _) in enumerate(results, 1):
                    key = (source, doc_idx)
                    if key not in rrf_scores:
                        rrf_scores[key] = 0
                    rrf_scores[key] += 1.0 / (self.rrf_k + rank)

            # Sort by RRF score and pick top N for re-ranking
            sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_candidate_n]

            if not sorted_keys:
                return []

            # Prepare candidates for re-ranking
            candidates = []
            for (source, doc_idx), rrf_score in sorted_keys:
                doc = self.stores[source]["vector_docs"][doc_idx]
                candidates.append({
                    "text": doc["text"],
                    "source": source,
                    "rrf_score": rrf_score,
                    "metadata": doc["metadata"]
                })

            # 2. Re-ranking (Stage 2)
            # Use Cross-Encoder to re-score candidates against original query
            with tracer.start_as_current_span("rerank") as rerank_span:
                rerank_span.set_attribute("candidate_count", len(candidates))
                pairs = [(query, c["text"]) for c in candidates]
                cross_scores = self.cross_encoder.predict(pairs)

                for i, score in enumerate(cross_scores):
                    candidates[i]["rerank_score"] = float(score)

                # Sort by re-ranked score
                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Build final output list with citations
            fused_results = []
            for doc in candidates[:top_k]:
                citation = self._format_citation(doc["metadata"], doc["source"])
                fused_results.append({
                    "text": doc["text"],
                    "citation": citation,
                    "source": doc["source"],
                    "rrf_score": doc["rrf_score"],
                    "rerank_score": doc["rerank_score"],
                    "metadata": doc["metadata"]
                })

            span.set_attribute("final_result_count", len(fused_results))
            return fused_results


