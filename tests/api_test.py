from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    # Use with TestClient to trigger startup/shutdown events if needed
    # But we want to avoid real initialization for fast tests
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("src.api.orchestrator")
def test_ask_endpoint(mock_orchestrator, client):
    """Test the /ask endpoint with a mocked orchestrator."""
    # Mock the orchestrator's response
    mock_orchestrator.generate_answer.return_value = {
        "answer": "Mocked answer about duty.",
        "sources": [
            {
                "text": "Perform your prescribed duty.",
                "citation": "Bhagavad Gita 2.47",
                "metadata": {},
            }
        ],
    }

    payload = {
        "query": "What does Krishna say about duty?",
        "religion": "bhagavad_gita",
        "top_k": 5,
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked answer about duty."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["citation"] == "Bhagavad Gita 2.47"

    # Verify the orchestrator was called correctly
    # Note: anyio.to_thread.run_sync calls it
    mock_orchestrator.generate_answer.assert_called_once_with(
        "What does Krishna say about duty?", "bhagavad_gita", 5
    )


def test_ask_endpoint_validation_error(client):
    """Test /ask endpoint with invalid payload."""
    payload = {
        "query": "",  # Pydantic might not catch empty string unless specified, but let's try missing field
        "top_k": -1,  # Should fail ge=1
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
