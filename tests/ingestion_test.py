import os
import shutil
import tempfile
import json
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock
from src.ingestion.parse_bible import parse_bible
from src.ingestion.parse_gita import parse_gita

class TestIngestion:
    @patch("src.ingestion.parse_bible.os.makedirs")
    @patch("src.ingestion.parse_bible.os.path.exists")
    @patch("src.ingestion.parse_bible.os.listdir")
    @patch("builtins.open")
    def test_parse_bible_mocked(self, mock_open, mock_listdir, mock_exists, mock_makedirs):
        mock_exists.return_value = True
        mock_listdir.return_value = ["01_Genesis.md"]
        
        bible_content = """# Genesis
## Chapter 1
1. In the beginning God created.
2. And the earth was without form.
"""
        # We'll capture calls to write instead of using StringIO for output
        output_mock = MagicMock()
        mock_open.side_effect = [
            StringIO(bible_content),
            output_mock
        ]
        
        # When used as a context manager
        output_mock.__enter__.return_value = output_mock
        
        parse_bible()
        
        # Check if write was called
        assert output_mock.write.called
        # Verify first record
        first_call = output_mock.write.call_args_list[0]
        data = json.loads(first_call[0][0])
        assert data["text"] == "In the beginning God created."
        assert data["metadata"]["book"] == "Genesis"

    @patch("src.ingestion.parse_gita.os.makedirs")
    @patch("src.ingestion.parse_gita.os.path.exists")
    @patch("src.ingestion.parse_gita.os.listdir")
    @patch("builtins.open")
    def test_parse_gita_mocked(self, mock_open, mock_listdir, mock_exists, mock_makedirs):
        mock_exists.side_effect = lambda p: True
        # listdir for raw_dir
        # listdir for folder_path
        mock_listdir.side_effect = [
            ["1"], # chapters
            ["1.md"] # verses in chapter 1
        ]
        
        gita_content = """### Translation:
Dhritarashtra said: O Sanjaya...
### Commentary:
..."""
        
        output_mock = MagicMock()
        mock_open.side_effect = [
            StringIO(gita_content),
            output_mock
        ]
        output_mock.__enter__.return_value = output_mock
        
        parse_gita()
        
        assert output_mock.write.called
        first_call = output_mock.write.call_args_list[0]
        data = json.loads(first_call[0][0])
        assert "Sanjaya" in data["text"]
        assert data["metadata"]["chapter"] == "1"
