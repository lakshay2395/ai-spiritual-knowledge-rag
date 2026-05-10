import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.hybrid_retriever import HybridRetriever

# Load environment variables
load_dotenv()

def setup_tracing():
    """
    Sets up observability using LangSmith and OpenTelemetry (Arize Phoenix).
    """
    # 1. LangSmith is handled automatically by LangChain if env vars are set
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("[LOG] LangSmith tracing enabled.")

    # 2. OpenTelemetry / Arize Phoenix setup
    otel_endpoint = os.getenv("PHOENIX_COLLECTOR_HTTP_ENDPOINT")
    if otel_endpoint:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            resource = Resource(attributes={
                "service.name": "ai-spiritual-knowledge-rag",
            })
            
            tracer_provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=f"{otel_endpoint}/v1/traces")
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(tracer_provider)

            # Instrument LangChain
            LangChainInstrumentor().instrument()
            print(f"[LOG] OpenTelemetry tracing enabled (Endpoint: {otel_endpoint})")
        except ImportError:
            print("[WARNING] OpenTelemetry/OpenInference dependencies not found. Skipping OTEL setup.")
        except Exception as e:
            print(f"[ERROR] Failed to setup OpenTelemetry tracing: {e}")

from opentelemetry import trace
import functools

# Initialize tracing
setup_tracing()

tracer = trace.get_tracer(__name__)

def traced(name=None):
    """Decorator to wrap a function in an OpenTelemetry span."""
    def decorator(func):
        span_name = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class Source(BaseModel):
    text: str = Field(description="The actual text content from the spiritual source")
    citation: str = Field(description="The book, chapter, and verse citation")
    metadata: Dict[str, Any] = Field(description="Additional metadata about the source")

class RAGResponse(BaseModel):
    answer: str = Field(description="The comprehensive answer derived strictly from the context")
    sources: List[Source] = Field(description="List of sources used to generate the answer")

class RAGOrchestrator:
    def __init__(self, retriever: Optional[HybridRetriever] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the RAGOrchestrator using LangChain components.
        """
        self.retriever = retriever or HybridRetriever()
        
        # 1. Initialize LLM
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True 
        )

        # 2. Setup Output Parser
        self.parser = PydanticOutputParser(pydantic_object=RAGResponse)

        # 3. Define Prompt Template
        self.system_prompt = (
            "You are a scholarly assistant specializing in spiritual texts.\n"
            "Your goal is to provide accurate, neutral answers strictly based on the provided context.\n\n"
            "RULES:\n"
            "1. Attempt to answer the question using the provided context.\n"
            "2. If the answer is not explicitly clear, provide the best possible interpretation based ONLY on the context, but clearly state if the evidence is limited.\n"
            "3. Always include citations for every claim in the format [Citation String]. Use the EXACT 'Source Citation' string provided in the context.\n"
            "4. Maintain a neutral, academic tone.\n"
            "5. Do not use outside knowledge.\n"
            "6. IMPORTANT: If there is any ambiguity or if the information is sparse, include a disclaimer that the interpretation may be limited and suggest the user consult with a religious scholar or spiritual leader for deeper understanding.\n\n"
            "{format_instructions}"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION: {query}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        # 4. Construct the Chain
        self.chain = self.prompt | self.llm | self.parser

    def _validate_citations(self, answer: str, sources: List[Source]) -> List[str]:
        """
        Extracts citations from the answer and verifies they exist in the sources.
        Returns a list of valid citation strings.
        """
        # Find all patterns like [Book Chapter:Verse] or [Bhagavad Gita 1.1]
        citations_in_text = re.findall(r'\[(.*?)\]', answer)
        valid_citations = {s.citation for s in sources}
        
        verified = [c for c in citations_in_text if c in valid_citations]
        return list(set(verified))

    def format_response(self, response_data: Dict[str, Any]) -> str:
        """
        Formats the RAGResponse into a beautiful CLI-friendly string with Source Cards.
        """
        answer = response_data.get("answer", "")
        sources = response_data.get("sources", [])
        
        output = [f"ANSWER:\n{answer}\n", "SOURCES:"]
        
        for i, source in enumerate(sources, 1):
            card = (
                f"  [{i}] {source['citation']}\n"
                f"      \"{source['text'][:150]}...\"\n"
            )
            output.append(card)
            
        return "\n".join(output)

    @traced("generate_answer")
    def generate_answer(self, query: str, religion: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Orchestrates the RAG process: Retrieve -> Validate -> Format.
        """
        span = trace.get_current_span()
        span.set_attribute("query", query)
        span.set_attribute("religion_filter", religion or "all")
        span.set_attribute("top_k", top_k)

        # 1. Retrieve
        print(f"[LOG] Retrieving context for query: '{query}' (Filter: {religion})")
        results = self.retriever.get_top_k(query, religion=religion, top_k=top_k)
        
        if not results:
            span.set_attribute("status", "no_results")
            return RAGResponse(
                answer=(
                    "I do not have enough information from the specific indexed texts to answer this accurately. "
                    "For a more complete understanding, I recommend consulting with a religious scholar or a spiritual leader."
                ),
                sources=[]
            ).model_dump()

        # 2. Format Context and Sources
        context_str, sources = self._prepare_context(results)
        span.set_attribute("context_length", len(context_str))

        # 3. Invoke Chain
        try:
            response = self.chain.invoke({
                "context": context_str,
                "query": query
            })
            
            # 4. Citation Validation
            final_response = self._validate_response(response, sources)
            
            span.set_attribute("status", "success")
            span.set_attribute("source_count", len(final_response.sources))
            return final_response.model_dump()
            
        except Exception as e:
            print(f"[ERROR] LangChain execution failed: {e}")
            span.record_exception(e)
            span.set_attribute("status", "error")
            return RAGResponse(
                answer="An error occurred while generating the answer.",
                sources=sources
            ).model_dump()

    def _prepare_context(self, results: List[Dict[str, Any]]) -> Tuple[str, List[Source]]:
        """Prepares the context string and Source objects from retrieval results."""
        context_blocks = []
        sources = []
        for res in results:
            context_blocks.append(f"Source Citation: {res['citation']}\nText: {res['text']}")
            sources.append(Source(
                text=res['text'],
                citation=res['citation'],
                metadata=res['metadata']
            ))
        return "\n---\n".join(context_blocks), sources

    def _validate_response(self, response: RAGResponse, sources: List[Source]) -> RAGResponse:
        """Validates citations in the response and filters unused sources."""
        valid_citations = self._validate_citations(response.answer, sources)
        response.sources = [s for s in sources if s.citation in valid_citations]
        
        # If no citations are found, keep all retrieved sources to be safe
        if not response.sources:
            response.sources = sources
        return response
