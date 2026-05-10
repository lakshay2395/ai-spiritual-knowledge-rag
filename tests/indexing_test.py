import os
import shutil
import tempfile
import json
import pytest
from src.indexing.keyword_store import KeywordStoreIndexer
from src.indexing.vector_store import VectorStoreIndexer


class TestIndexing:
    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        data = [
            {
                "text": "In the beginning God created the heaven and the earth.",
                "metadata": {"book": "Genesis", "chapter": "1", "verse": "1"},
            },
            {
                "text": "And the earth was without form, and void.",
                "metadata": {"book": "Genesis", "chapter": "1", "verse": "2"},
            },
        ]
        with open(os.path.join(temp_dir, "test.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_keyword_indexing(self, temp_data_dir):
        indexer = KeywordStoreIndexer()
        data = indexer.load_data(temp_data_dir)
        assert len(data) == 2

        indexer.create_index(data)
        assert indexer.bm25 is not None

        with tempfile.TemporaryDirectory() as output_dir:
            indexer.save_index(output_dir, "test_index")
            assert os.path.exists(os.path.join(output_dir, "test_index_bm25.pkl"))
            assert os.path.exists(os.path.join(output_dir, "test_index_docs.pkl"))

    def test_vector_indexing(self, temp_data_dir):
        # Use a very small model for testing if possible, or just mock it.
        # MiniLM is already small.
        indexer = VectorStoreIndexer()
        data = indexer.load_data(temp_data_dir)
        assert len(data) == 2

        indexer.create_index(data)
        assert indexer.index is not None
        assert indexer.index.ntotal == 2

        with tempfile.TemporaryDirectory() as output_dir:
            indexer.save_index(output_dir, "test_vector")
            assert os.path.exists(os.path.join(output_dir, "test_vector.faiss"))
            assert os.path.exists(os.path.join(output_dir, "test_vector_docs.pkl"))

    def test_preprocess(self):
        indexer = KeywordStoreIndexer()
        tokens = indexer.preprocess("The Quick Brown Fox!")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens  # stop word
