import os
import re
from typing import List, Dict, Any, Optional
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

# Initialize tracing
setup_tracing()

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

    def generate_answer(self, query: str, religion: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Orchestrates the RAG process: Retrieve -> Validate -> Format.
        """
        # 1. Retrieve
        print(f"[LOG] Retrieving context for query: '{query}' (Filter: {religion})")
        results = self.retriever.get_top_k(query, religion=religion, top_k=top_k)
        
        if not results:
            return RAGResponse(
                answer=(
                    "I do not have enough information from the specific indexed texts to answer this accurately. "
                    "For a more complete understanding, I recommend consulting with a religious scholar or a spiritual leader."
                ),
                sources=[]
            ).model_dump()

        # 2. Format Context and Sources
        context_blocks = []
        sources = []
        for res in results:
            context_blocks.append(f"Source Citation: {res['citation']}\nText: {res['text']}")
            sources.append(Source(
                text=res['text'],
                citation=res['citation'],
                metadata=res['metadata']
            ))
            
        context_str = "\n---\n".join(context_blocks)

        # 3. Invoke Chain
        try:
            response = self.chain.invoke({
                "context": context_str,
                "query": query
            })
            
            # 4. Citation Validation
            # Filter the sources to only those actually cited in the answer
            valid_citations = self._validate_citations(response.answer, sources)
            response.sources = [s for s in sources if s.citation in valid_citations]
            
            # If the LLM failed to cite but we have results, we keep the retrieved sources 
            # as 'potential sources' or just keep them all if validation is empty?
            # Scholarly rigour: if it's not cited, it shouldn't be in the list?
            # For now, if no citations are found, we keep all retrieved sources to be safe, 
            # but mark them as potentially unused.
            if not response.sources:
                response.sources = sources

            return response.model_dump()
            
        except Exception as e:
            print(f"[ERROR] LangChain execution failed: {e}")
            return RAGResponse(
                answer="An error occurred while generating the answer.",
                sources=sources
            ).model_dump()
