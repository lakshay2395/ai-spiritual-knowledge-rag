I am in Week 2 (The RAG Pipeline) of building my AI Spiritual Knowledge Engine. I have a functional HybridRetriever in src/retrieval/hybrid_retriever.py that returns a list of fused results (text + metadata) using Reciprocal Rank Fusion (RRF).

Please write a professional, modular Python script for src/rag/orchestrator.py that accomplishes the following:

1.  Class-Based Structure: Create a RAGOrchestrator class.
2.  Initialization:
* Initialize the HybridRetriever to handle document fetching.
* Setup an LLM interface (use a placeholder for gemini flash only).
3.  The "Scholarly Neutral" System Prompt:
* Define a system prompt that enforces strict grounding. The AI must only answer using the provided context.
* If the answer is not in the context, it must state: 'I do not have enough information from the texts to answer this'.
* Include instructions to always include citations in the format [Book Chapter:Verse].
4.  Context Injection Logic:
* Implement a generate_answer method that takes a user query.
* Retrieve the top 3-5 fused results from the HybridRetriever.
* Format these results into a single string to be injected into the LLM prompt as 'Context'.
5.  Output Handling:
* The method should return a JSON-like dictionary containing the answer and a list of sources (metadata).
6.  Requirements:
* Use Pydantic for response schema validation if possible.
* Include error handling for empty retrieval results.