import os
import sys
import pickle

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from src.retrieval.hybrid_retriever import HybridRetriever

class HybridSearchTester:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the tester, loads models, and prepares NLTK.
        """
        print("Initializing HybridSearchTester...")
        self.model = SentenceTransformer(model_name)
        
        # Ensure NLTK resources for keyword preprocessing
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('tokenizers/punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # Storage for indices and document stores
        self.vector_indices = {}  # {source_name: faiss_index}
        self.vector_docs = {}     # {source_name: docs_list}
        self.keyword_indices = {} # {source_name: bm25_object}
        self.keyword_docs = {}    # {source_name: docs_list}

    def load_indices(self, vector_dir: str, keyword_dir: str):
        """
        Loads all available indices from the specified directories.
        """
        # Load Vector Indices
        print(f"Loading vector indices from {vector_dir}...")
        for file in os.listdir(vector_dir):
            if file.endswith(".faiss"):
                name = file.replace(".faiss", "")
                self.vector_indices[name] = faiss.read_index(os.path.join(vector_dir, file))
                
                docs_path = os.path.join(vector_dir, f"{name}_docs.pkl")
                if os.path.exists(docs_path):
                    with open(docs_path, "rb") as f:
                        self.vector_docs[name] = pickle.load(f)
                print(f" - Loaded vector store: {name}")

        # Load Keyword Indices
        print(f"Loading keyword indices from {keyword_dir}...")
        for file in os.listdir(keyword_dir):
            if file.endswith("_bm25.pkl"):
                name = file.replace("_bm25.pkl", "")
                with open(os.path.join(keyword_dir, file), "rb") as f:
                    self.keyword_indices[name] = pickle.load(f)
                
                docs_path = os.path.join(keyword_dir, f"{name}_docs.pkl")
                if os.path.exists(docs_path):
                    with open(docs_path, "rb") as f:
                        self.keyword_docs[name] = pickle.load(f)
                print(f" - Loaded keyword store: {name}")

    def preprocess_query(self, query: str) -> List[str]:
        """
        Tokenizes and cleans the query for keyword search.
        """
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s]', '', query)
        tokens = word_tokenize(query)
        return [t for t in tokens if t not in self.stop_words]

    def semantic_search(self, query: str, source: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top results using semantic (vector) search.
        """
        if source not in self.vector_indices:
            print(f"Warning: Source '{source}' not found in vector indices.")
            return []

        # Generate query embedding
        query_vector = self.model.encode([query]).astype("float32")
        
        # Search FAISS index
        distances, indices = self.vector_indices[source].search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.vector_docs[source]):
                doc = self.vector_docs[source][idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(dist),
                    "method": "Semantic"
                })
        return results

    def keyword_search(self, query: str, source: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top results using keyword (BM25) search.
        """
        if source not in self.keyword_indices:
            print(f"Warning: Source '{source}' not found in keyword indices.")
            return []

        # Tokenize query
        tokenized_query = self.preprocess_query(query)
        
        # Get BM25 scores
        scores = self.keyword_indices[source].get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Only return matches
                doc = self.keyword_docs[source][idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(scores[idx]),
                    "method": "Keyword"
                })
        return results

    def display_results(self, results: List[Dict[str, Any]], source: str):
        """
        Prints formatted results to the terminal.
        """
        if not results:
            print(f"No results found for {source}.")
            return

        for i, res in enumerate(results, 1):
            meta = res['metadata']
            # Format citation based on source type
            if 'book' in meta:
                citation = f"{meta['book']} {meta['chapter']}:{meta['verse']}"
            else:
                citation = f"Gita {meta['chapter']}.{meta['verse']}"
            
            snippet = res['text'][:150].replace('\n', ' ') + "..." if len(res['text']) > 150 else res['text']
            
            print(f"{i}. [{citation}] | Score: {res['score']:.4f} | Method: {res['method']}")
            print(f"   Text: {snippet}\n")

if __name__ == "__main__":
    tester = HybridSearchTester()
    
    # Paths from Phase 1
    VECTOR_DIR = "data/indices/vector"
    KEYWORD_DIR = "data/indices/keyword"
    
    try:
        tester.load_indices(VECTOR_DIR, KEYWORD_DIR)
        
        print("\n" + "="*50)
        print("TEST CASE 1: Keyword Search (Bible)")
        print("Query: 'Melchizedek'")
        print("="*50)
        bible_kw_results = tester.keyword_search("Melchizedek", "bible")
        tester.display_results(bible_kw_results, "Bible")

        print("\n" + "="*50)
        print("TEST CASE 1: Semantic Search (Bible)")
        print("Query: 'Melchizedek'")
        print("="*50)
        bible_kw_results = tester.semantic_search("Melchizedek", "bible")
        tester.display_results(bible_kw_results, "Bible")

        # --- Phase 3 Test Case ---
        print("\n" + "="*50)
        print("PHASE 3: Hybrid Retrieval with RRF + Re-ranking")
        print("Query: 'What does the text say about eternal life?'")
        print("="*50)
        retriever = HybridRetriever()
        fused_results = retriever.get_top_k("What does the text say about eternal life?", top_k=5)
        
        for i, res in enumerate(fused_results, 1):
            print(f"{i}. [{res['citation']}] (RRF: {res['rrf_score']:.5f}, Re-rank: {res['rerank_score']:.4f})")
            print(f"   {res['text'][:150]}...\n")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
