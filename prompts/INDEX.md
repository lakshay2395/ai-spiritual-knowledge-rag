I have a structured jsonl files under two subfolders: 

1. In files in `data/processed/bhagavad-gita-as-it-is`, jsonl files have a sample like this : 
{"text": "Dhritarashtra said: O Sanjaya, after my sons and the sons of Pandu assembled in the place of pilgrimage at Kurukshetra, desiring to fight, what did they do?", "metadata": {"chapter": "1", "verse": "1"}}

2. In files in `data/processed/mdbible`, jsonl files have a sample like this : 
{"text": "In the beginning, God created the heavens and the earth.", "metadata": {"book": "Genesis", "chapter": "1", "verse": "1"}}

Please write two professional, modular Python scripts:

src/indexing/vector_store.py:
- Use sentence-transformers/all-MiniLM-L6-v2 to generate embeddings.
- Use FAISS (IndexFlatL2) to create a vector index.
- Implement logic to batch-process the JSONL and save the .faiss index and a corresponding metadata .pkl file to data/indices/.

src/indexing/keyword_store.py:
- Use rank-bm25 to build a keyword index.
- Include a basic preprocessing function (lowercase, stop-word removal, and tokenization).
- Save the trained BM25 object as a .pkl file to data/indices/.

Requirements: 
- Both scripts must follow a class-based structure.
- Include error handling for missing directories or files.
- Ensure the metadata is preserved and linked to the index results.