I am in Phase 2 of building my AI Spiritual Knowledge Engine. I have successfully created two sets of index files under data/indices/:

Semantic (FAISS) [data/indices/vector]: bhagavad_gita.faiss / bhagavad_gita_docs.pkl and bible.faiss / bible_docs.pkl.

Keyword (BM25) [data/indices/keyword]: bhagavad_gita_bm25.pkl / bhagavad_gita_docs.pkl and bible_bm25.pkl / bible_docs.pkl.

Please write a professional Python script for src/retrieval/search_test.py that accomplishes the following:

1.  Class-Based Structure: Create a HybridSearchTester class.
2.  Initialization:
* Load the all-MiniLM-L6-v2 model for generating query embeddings.
* Load both FAISS indices and their corresponding *_docs.pkl files.
* Load both BM25 objects and their corresponding *_docs.pkl files.
3.  Search Logic:
* Semantic Method: Implement a search that takes a query, generates an embedding (casting it to float32), and retrieves the top 3 results from the FAISS index.
* Keyword Method: Implement a search that tokenizes a query and retrieves the top 3 results from the BM25 index.
4.  One-Stop Retrieval: Use Positional Alignment to ensure that when an index returns an ID (e.g., 42), the script immediately pulls the full document (text + metadata) from the corresponding _docs.pkl file.
5.  Output Formatting: Print results to the terminal with clear labels showing: [Source Citation], Score, and the first 150 characters of the Verse Text.
6.  Test Cases: In the if __name__ == '__main__': block, include sample queries for:
* 'Melchizedek' (Keyword test for Bible).
* 'What is the nature of the soul?' (Semantic test for Gita) .

Requirements: Ensure the script handles the logic for both the Bible and Bhagavad Gita indices separately within the same test flow