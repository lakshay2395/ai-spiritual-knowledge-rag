import nltk
from sentence_transformers import CrossEncoder, SentenceTransformer


def download_assets():
    print("Downloading NLTK data...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    print("Downloading SentenceTransformer model...")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Downloading CrossEncoder model...")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


if __name__ == "__main__":
    download_assets()
