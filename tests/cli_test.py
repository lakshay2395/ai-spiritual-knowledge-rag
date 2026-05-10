import pytest
from click.testing import CliRunner
from src.cli import cli, display_welcome, display_response
from unittest.mock import patch, MagicMock

def test_display_welcome():
    with patch("src.cli.console.print") as mock_print:
        display_welcome()
        assert mock_print.called

def test_display_response():
    response = {
        "answer": "The soul is eternal.",
        "sources": [
            {"text": "Some text", "citation": "Gita 2.20", "metadata": {}}
        ]
    }
    with patch("src.cli.console.print") as mock_print:
        display_response("What is the soul?", response)
        assert mock_print.called

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CLI for the AI Spiritual Knowledge Engine" in result.output

@patch("src.cli.RAGOrchestrator")
@patch("src.cli.HybridRetriever")
def test_cli_ask_query(mock_retriever, mock_orch_class):
    mock_orch = MagicMock()
    mock_orch_class.return_value = mock_orch
    mock_orch.generate_answer.return_value = {
        "answer": "Test answer",
        "sources": []
    }
    
    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "--query", "What is life?"])
    assert result.exit_code == 0
    assert "Test answer" in result.output
