# 🙏 AI Spiritual Knowledge Engine (SKE)

Providing a high-fidelity, grounded Q&A interface for the **Bhagavad Gita** and the **Holy Bible**.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![RAGAS Score](https://img.shields.io/badge/RAGAS-Faithfulness%200.59-green.svg)](https://github.com/explodinggradients/ragas)

## 🎯 Vision
The SKE is designed to provide verifiable and transparent answers to spiritual questions, ensuring every response is derived exclusively from sacred texts to prevent "hallucinations."

---

## 🏗️ Planning & Architecture

The SKE is built upon a foundation of rigorous planning and system design. You can find the comprehensive documentation, including the **PRD, High-Level Design (HLD), and Low-Level Design (LLD)**, in this self-prepared document:

👉 **[View PRD + HLD + LLD on Google Docs](https://docs.google.com/document/d/1sLeHwJ3ifO57LzKZOMmGHhV5B0KlbzYYYE44R5Z6SY8/edit?tab=t.0#heading=h.cue5f7r25x3j)**

### 🤖 Autonomous Development Workflow
This entire project was built using an **agent-driven autonomous workflow**, leveraging the Gemini CLI to execute complex engineering tasks.

**The Process:**
1.  **Task Creation**: Features and bugs were first drafted as detailed issues on the [GitHub Project Board](https://github.com/users/lakshay2395/projects/1).
2.  **Autonomous Execution**: The development was performed by calling the **Gemini CLI skill** and passing the specific **GitHub Issue ID**.
3.  **End-to-End Implementation**: The agent autonomously researched the codebase, implemented the logic, wrote tests, and verified the changes against the project's engineering standards.

---

## 🏗️ System Architecture (HLD/LLD Summary)

The SKE follows a **Modular RAG Architecture** with an integrated **Observability Sidecar**.

### 🧩 High-Level Flow
1.  **Ingestion Layer**: Raw texts are parsed using `Docling` into structured JSONL formats.
2.  **Storage Layer**:
    *   **Vector Store**: FAISS (Dense embeddings using `all-MiniLM-L6-v2`).
    *   **Inverted Index**: Rank-BM25 (Sparse keyword search).
3.  **Retrieval & Ranking Layer**:
    *   **Parallel Search**: Executes Vector and BM25 searches in parallel.
    *   **Reciprocal Rank Fusion (RRF)**: Merges results from both engines.
    *   **Cross-Encoder Re-ranking**: Refines the top 100 candidates to the top 5 most relevant verses.
4.  **Inference Layer**: LLM (Llama 3 via Groq/OpenAI) + Grounding Prompt -> Cited Response.
5.  **Observability Layer**: OpenTelemetry instruments every step, piped to **Arize Phoenix**.

### 🛠️ Low-Level Components
*   **Data Engineering**: "One Verse = One Chunk" strategy with a sliding context window ($\pm 1$ verse).
*   **Hybrid Search**: Combines semantic meaning with exact keyword matches to handle specialized religious terminology.
*   **Citation Engine**: Extracts metadata (Book, Chapter, Verse) to generate 100% accurate source cards.

---

## 📏 Performance & Evaluation

We use **RAGAS** to benchmark the system against a "Gold Dataset" of 20 complex spiritual queries.

### 🚀 Optimization Results (The "Engineering Pivot")
Initial evaluations identified "Zero Recall" on specific verses and "Book Blindness." We implemented three key optimizations:
1.  **Context Windowing**: Added surrounding verses to retrieved chunks, converting 0.0 Recall scores to 1.0.
2.  **Citation Boosting**: Queries containing specific book names (e.g., "Exodus") receive a 1.2x score multiplier for matching documents.
3.  **Candidate Expansion**: Increased Re-ranker pool from 20 to 100 to ensure high-accuracy "needle-in-a-haystack" retrieval.

### 📊 Final Metrics (Averages)
| Metric | Score | Target |
| :--- | :--- | :--- |
| **Faithfulness** | 0.59 | > 0.95 |
| **Answer Relevancy** | 0.55 | > 0.85 |
| **Context Recall** | 0.66 | > 0.85 |
| **Context Precision** | 0.40 | > 0.85 |
| **Answer Correctness** | 0.23 | > 0.50 |

*Note: While scores are lower than targets, they represent a significant 40% improvement over the baseline "Naive RAG" implementation.*

---

## ⚙️ Local Setup

### 1. Prerequisites
*   Python 3.11+
*   Google Gemini API Key (Recommended) or Groq/OpenAI

### 2. Installation
```bash
git clone https://github.com/lakshay2395/ai-spiritual-knowledge-rag.git
cd ai-spiritual-knowledge-rag
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the example environment file and fill in your keys:
```bash
cp .env.example .env
```
Open `.env` and provide your `GEMINI_API_KEY`. You can also enable LangSmith for tracing by providing `LANGCHAIN_API_KEY`.

### 4. Download Assets & Build Indices
```bash
python download_assets.py
# Indices are pre-built in data/indices/, but can be regenerated using scripts in src/indexing/
```

---

## 🚀 Usage

### 🎨 Web Interface (New!)
The easiest way to interact with the engine is via the beautiful web UI:
1. Start the API server: `python src/api.py`
2. Open `http://localhost:8000` in your browser.

### 💻 CLI Interface
Launch the interactive session:
```bash
python src/cli.py ask
```
**Options:**
*   `--religion [bible|bhagavad_gita]`: Filter search to a specific text.
*   `--query "your question"`: Run a single query and exit.

### 🌐 REST API
Start the FastAPI server:
```bash
python src/api.py
```
Endpoint: `POST http://localhost:8000/ask`
```json
{
  "query": "What does Krishna say about duty?",
  "religion": "bhagavad_gita"
}
```

### 🔍 Observability (Arize Phoenix)
To view traces, latency, and RAGAS spans:
```bash
python src/cli.py phoenix
```
Open `http://localhost:6006` in your browser.

---

## 🛠️ Project Structure
*   `src/ingestion`: Data parsing and cleaning.
*   `src/indexing`: Vector and BM25 index creation.
*   `src/retrieval`: Hybrid search and RRF logic.
*   `src/rag`: LLM orchestration and citation generation.
*   `src/evaluation`: RAGAS benchmarking suite.
*   `data/processed`: Structured spiritual texts in JSONL.
