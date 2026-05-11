import os

import pandas as pd


def generate_summary(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    metrics = [m for m in metrics if m in df.columns]

    print("=" * 50)
    print(f"EVALUATION SUMMARY: {os.path.basename(file_path)}")
    print("=" * 50)

    summary_stats = df[metrics].describe().loc[["mean", "min", "max"]]
    print(summary_stats.to_string())

    print("\n" + "=" * 50)
    print("TOP 3 BEST PERFORMING QUERIES (by Average Score)")
    df["avg_score"] = df[metrics].mean(axis=1)
    top_3 = df.nlargest(3, "avg_score")[["user_input", "avg_score"]]
    for _, row in top_3.iterrows():
        print(f"- {row['user_input']} (Score: {row['avg_score']:.3f})")

    print("\n" + "=" * 50)
    print("TOP 3 WEAKEST QUERIES (by Average Score)")
    bottom_3 = df.nsmallest(3, "avg_score")[["user_input", "avg_score"]]
    for _, row in bottom_3.iterrows():
        print(f"- {row['user_input']} (Score: {row['avg_score']:.3f})")
    print("=" * 50)


if __name__ == "__main__":
    generate_summary("data/evaluation/ragas_report_optimized.csv")
