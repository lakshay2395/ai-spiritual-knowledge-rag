import json
import os
import pickle
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStoreIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the VectorStoreIndexer with a sentence-transformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def load_data(self, directory: str) -> List[Dict[str, Any]]:
        """
        Loads all JSONL files from a directory.
        """
        data = []
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line))
        return data

    def create_index(self, data: List[Dict[str, Any]], batch_size: int = 64):
        """
        Processes data in batches to generate embeddings and builds a FAISS index.
        """
        texts = [item["text"] for item in data]
        # Store full document (text + metadata) for retrieval
        self.documents = data

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} items...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True
        )
        embeddings = np.array(embeddings).astype("float32")

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print("FAISS index built successfully.")

    def save_index(self, output_dir: str, index_name: str):
        """
        Saves the FAISS index and full documents to the specified directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index_path = os.path.join(output_dir, f"{index_name}.faiss")
        docs_path = os.path.join(output_dir, f"{index_name}_docs.pkl")

        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Index and documents saved to {output_dir}")


if __name__ == "__main__":
    # Example usage for Gita and Bible
    indexer = VectorStoreIndexer()

    # Process Bhagavad Gita
    try:
        gita_data = indexer.load_data("data/processed/bhagavad-gita-as-it-is")
        indexer.create_index(gita_data)
        indexer.save_index("data/indices/vector", "bhagavad_gita")
    except Exception as e:
        print(f"Error processing Gita: {e}")

    # Process Bible
    try:
        bible_data = indexer.load_data("data/processed/mdbible")
        indexer.create_index(bible_data)
        indexer.save_index("data/indices/vector", "bible")
    except Exception as e:
        print(f"Error processing Bible: {e}")
