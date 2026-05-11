import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import anyio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.orchestrator import RAGOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spiritual-rag-api")

# Global orchestrator instance
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes the RAG Orchestrator once on startup.
    """
    global orchestrator
    logger.info("Initializing RAG Orchestrator...")
    orchestrator = RAGOrchestrator()
    logger.info("RAG Orchestrator initialized.")
    yield
    logger.info("Shutting down RAG Orchestrator...")
    # Add any cleanup logic for orchestrator if necessary


app = FastAPI(
    title="AI Spiritual Knowledge RAG API",
    description="REST API for querying the Bhagavad Gita and the Bible using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class SourceModel(BaseModel):
    text: str
    citation: str
    metadata: dict


class QueryRequest(BaseModel):
    query: str = Field(
        ..., json_schema_extra={"example": "What does Krishna say about duty?"}
    )
    religion: Optional[str] = Field(
        None, json_schema_extra={"example": "bhagavad_gita"}
    )
    top_k: int = Field(10, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceModel]


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API and engine are running."""
    return {"status": "healthy", "orchestrator_loaded": orchestrator is not None}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Asynchronous endpoint to query the spiritual RAG engine.
    Runs the engine's inference in a thread pool to avoid blocking the event loop.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    logger.info(f"Query: {request.query} | Religion: {request.religion}")

    try:
        # Run the blocking orchestrator call in a thread pool
        response = await anyio.to_thread.run_sync(
            orchestrator.generate_answer, request.query, request.religion, request.top_k
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing the query.",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
