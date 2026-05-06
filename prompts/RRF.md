I am moving into Phase 3: Retrieval & Ranking for my AI Spiritual Knowledge Engine. I have verified that my BM25 and FAISS indices are saved in data/indices/keyword and data/indices/vector, using the Encapsulated Storage model (where *_docs.pkl contains both text and metadata).

Please write a professional, modular Python script for src/retrieval/hybrid_retriever.py that includes:Class HybridRetriever:

* Initialization: Load the all-MiniLM-L6-v2 embedding model. Load the FAISS indices and BM25 objects for both the Bible and Bhagavad Gita, along with their respective {name}_docs.pkl files.Method get_top_k:
* Accept a query string and an optional religion filter.
* Run a Semantic Search (FAISS) and a Keyword Search (BM25) in parallel for the relevant indices.  
* Reciprocal Rank Fusion Implementation:
    * Implement the Reciprocal Rank Fusion algorithm to merge the two ranked lists.
    * Use the industry-standard constant k=60.
    * Ensure the math uses document ranks, not raw scores, to bridge the scale gap between FAISS and BM25.  
* One-Stop Retrieval:
    * Use the fused ranks to pull the full document objects (text, book, chapter, verse) from the {name}_docs.pkl files.
    * Output: Return a sorted list of the top 5 unique "fused" results, each containing the full text and its citation.  

Requirements:
* Include error handling for missing index files.
* Add a main block to test the retriever with a query like 'What does the text say about eternal life?' to verify the fusion works across both engines."