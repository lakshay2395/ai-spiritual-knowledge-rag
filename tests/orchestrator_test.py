import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.orchestrator import RAGOrchestrator


def test_rag_orchestration():
    """
    Tests the full RAG pipeline: Retrieval -> Generation.
    """
    print("\n" + "=" * 50)
    print("TESTING RAG ORCHESTRATOR")
    print("=" * 50)

    try:
        orchestrator = RAGOrchestrator()
        test_query = "What does the text say about the nature of the soul?"

        print(f"Query: '{test_query}'")
        result = orchestrator.generate_answer(test_query)

        print("\n=== FINAL RAG OUTPUT ===")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error during orchestration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_rag_orchestration()
