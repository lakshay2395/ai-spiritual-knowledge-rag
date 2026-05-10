import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retrieval.hybrid_retriever import HybridRetriever

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
            convert_system_message_to_human=True # Useful for some Gemini versions
        )

        # 2. Setup Output Parser
        self.parser = PydanticOutputParser(pydantic_object=RAGResponse)

        # 3. Define Prompt Template
        self.system_prompt = (
            "You are a scholarly assistant specializing in spiritual texts.\n"
            "Your goal is to provide accurate, neutral answers strictly based on the provided context.\n\n"
            "RULES:\n"
            "1. ONLY use the provided context to answer the question.\n"
            "2. If the answer is not in the context, state: 'I do not have enough information from the texts to answer this'.\n"
            "3. Always include citations for every claim in the format [Book Chapter:Verse].\n"
            "4. Maintain a neutral, academic tone.\n"
            "5. Do not use outside knowledge.\n\n"
            "{format_instructions}"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION: {query}")
        ]).partial(format_instructions=self.parser.get_format_instructions())

        # 4. Construct the Chain
        # Note: We handle retrieval manually to maintain the same interface and logic
        self.chain = self.prompt | self.llm | self.parser

    def generate_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Orchestrates the RAG process using LangChain: Retrieve -> Map -> Chain Invoke.
        """
        # 1. Retrieve
        print(f"[LOG] Retrieving context for query: '{query}'")
        results = self.retriever.get_top_k(query, top_k=top_k)
        
        if not results:
            print("[LOG] No results retrieved.")
            return RAGResponse(
                answer="I do not have enough information from the texts to answer this",
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
            # LangChain returns a RAGResponse object because of the PydanticOutputParser
            response = self.chain.invoke({
                "context": context_str,
                "query": query
            })
            
            # Ensure sources from retrieval are injected back or correctly parsed
            # Sometimes LLMs might hallucinate sources if asked to return them in JSON.
            # We override or merge with our verified retrieval sources for integrity.
            response.sources = sources 
            
            return response.model_dump()
            
        except Exception as e:
            print(f"[ERROR] LangChain execution failed: {e}")
            return RAGResponse(
                answer="An error occurred while generating the answer.",
                sources=sources
            ).model_dump()
