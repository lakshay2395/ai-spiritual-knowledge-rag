import json
import os
import pickle
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class KeywordStoreIndexer:
    def __init__(self):
        """
        Initializes the KeywordStoreIndexer and ensures NLTK resources are available.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab') 
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.bm25 = None
        self.metadata = []

    def preprocess(self, text: str) -> List[str]:
        """
        Basic preprocessing: lowercase, remove non-alphanumeric, tokenize, and remove stop-words.
        """
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop-words
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

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

    def create_index(self, data: List[Dict[str, Any]]):
        """
        Preprocesses the corpus and builds the BM25 index.
        """
        texts = [item["text"] for item in data]
        # Store full document (text + metadata) for retrieval
        self.documents = data 
        
        print(f"Preprocessing {len(texts)} items for BM25...")
        tokenized_corpus = [self.preprocess(text) for text in texts]
        
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully.")

    def save_index(self, output_dir: str, index_name: str):
        """
        Saves the BM25 object and full documents to the specified directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        index_path = os.path.join(output_dir, f"{index_name}_bm25.pkl")
        docs_path = os.path.join(output_dir, f"{index_name}_docs.pkl")
        
        with open(index_path, "wb") as f:
            pickle.dump(self.bm25, f)
            
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"BM25 index and documents saved to {output_dir}")

if __name__ == "__main__":
    indexer = KeywordStoreIndexer()
    
    # Process Bhagavad Gita
    try:
        gita_data = indexer.load_data("data/processed/bhagavad-gita-as-it-is")
        indexer.create_index(gita_data)
        indexer.save_index("data/indices/keyword", "bhagavad_gita")
    except Exception as e:
        print(f"Error processing Gita: {e}")
        
    # Process Bible
    try:
        bible_data = indexer.load_data("data/processed/mdbible")
        indexer.create_index(bible_data)
        indexer.save_index("data/indices/keyword", "bible")
    except Exception as e:
        print(f"Error processing Bible: {e}")
