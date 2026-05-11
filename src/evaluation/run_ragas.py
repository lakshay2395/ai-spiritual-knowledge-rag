import argparse
import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import (AnswerCorrectness, AnswerRelevancy,
                           ContextPrecision, ContextRecall, Faithfulness)

# Add project root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.rag.orchestrator import RAGOrchestrator

# Load environment variables
load_dotenv()


def run_evaluation(
    limit: int = None, output_path: str = "data/evaluation/ragas_report_optimized.csv"
):
    # 1. Load Gold Dataset
    gold_path = "data/evaluation/gold_dataset.json"
    if not os.path.exists(gold_path):
        print(f"Error: {gold_path} not found.")
        return

    with open(gold_path, "r") as f:
        gold_data = json.load(f)

    if limit:
        gold_data = gold_data[:limit]

    # Check for API Key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "CRITICAL ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not found. Please set it in your environment or .env file."
        )
        return

    orchestrator = RAGOrchestrator(model_name="gemini-2.5-flash")

    evaluation_results = []

    print(f"Running evaluation on {len(gold_data)} samples...")

    for i, sample in enumerate(gold_data):
        query = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"[{i+1}/{len(gold_data)}] Processing query: '{query[:50]}...'")

        # Run RAG pipeline
        try:
            result = orchestrator.generate_answer(query)

            answer = result.get("answer", "")
            contexts = [s["text"] for s in result.get("sources", [])]

            evaluation_results.append(
                {
                    "user_input": query,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": ground_truth,
                }
            )
        except Exception as e:
            print(f"  [ERROR] Failed to process query: {e}")

    if not evaluation_results:
        print("No evaluation results were generated. Skipping RAGAS evaluation.")
        return

    # 2. Convert to RAGAS EvaluationDataset
    dataset = EvaluationDataset.from_list(evaluation_results)

    # 3. Setup RAGAS with Gemini
    print("Setting up RAGAS with Gemini LLM and Embeddings...")
    client = genai.Client(api_key=api_key)
    llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    # Use LangChain embeddings for better compatibility
    lc_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", google_api_key=api_key
    )
    embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    # Instantiate metrics
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    for i, m in enumerate(metrics):
        print(f"Metric {i}: {type(m)}")

    # 4. Run Evaluation
    print("Computing RAGAS metrics (this may take a while)...")
    try:
        result = evaluate(dataset, metrics=metrics)

        # 5. Save and Print Report
        df = result.to_pandas()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\nEvaluation Complete!")
        print(f"Detailed report saved to {output_path}")
        print("\nSummary Metrics:")
        print(result)
    except Exception as e:
        print(f"CRITICAL ERROR during RAGAS evaluation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the spiritual RAG pipeline."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/ragas_report.csv",
        help="Path to save the evaluation report.",
    )
    args = parser.parse_args()

    run_evaluation(limit=args.limit, output_path=args.output)
