import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Set page config
st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")


def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df


def main():
    st.title("📊 RAG Evaluation Dashboard")
    st.markdown(
        """
    This dashboard visualizes the evaluation results of the Spiritual Knowledge RAG system using **RAGAS** metrics.
    """
    )

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    report_file = st.sidebar.selectbox(
        "Select Report File",
        [
            "data/evaluation/ragas_report_optimized.csv",
            "data/evaluation/ragas_report.csv",
        ],
    )

    df = load_data(report_file)

    if df is None:
        st.error(f"Report file not found: {report_file}")
        return

    # Metric columns
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]
    # Filter out metrics not in dataframe
    metrics = [m for m in metrics if m in df.columns]

    # Overall Metrics Summary
    st.header("📈 Overall Performance")
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        avg_score = df[metric].mean()
        cols[i].metric(label=metric.replace("_", " ").title(), value=f"{avg_score:.3f}")

    # Charts Section
    st.header("📊 Detailed Analysis")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Score Distributions")
        selected_metric = st.selectbox("Select Metric for Distribution", metrics)
        fig_dist = px.histogram(
            df,
            x=selected_metric,
            nbins=20,
            title=f"Distribution of {selected_metric.replace('_', ' ').title()}",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.subheader("Metric Correlations")
        metric_x = st.selectbox("Metric X", metrics, index=0)
        metric_y = st.selectbox("Metric Y", metrics, index=1 if len(metrics) > 1 else 0)
        fig_scatter = px.scatter(
            df,
            x=metric_x,
            y=metric_y,
            hover_data=["user_input"],
            title=f"{metric_x.title()} vs {metric_y.title()}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Comparison Chart
    st.subheader("Metric Averages")
    avg_df = df[metrics].mean().reset_index()
    avg_df.columns = ["Metric", "Average Score"]
    fig_bar = px.bar(
        avg_df,
        x="Metric",
        y="Average Score",
        color="Metric",
        title="Average RAGAS Scores",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Data Table
    st.header("📄 Detailed Evaluation Data")

    # Search/Filter
    search_query = st.text_input("Search in User Input")
    if search_query:
        df_display = df[
            df["user_input"].str.contains(search_query, case=False, na=False)
        ]
    else:
        df_display = df

    st.dataframe(df_display, use_container_width=True)

    # Option to download data
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ragas_evaluation_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
