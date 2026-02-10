import os, pickle
import requests, hashlib
from typing import TypedDict, Annotated, List, Sequence
from operator import add
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.base import Embeddings

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()
# ============================================
# CONFIGURATION - Load from environment
# ============================================

BASE_URL = os.getenv("BASE_URL")
OPENAI_COMPAT_URL = os.getenv("OPENAI_COMPAT_URL")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
API_KEY = os.getenv("API_KEY")


# ============================================
# 1. DEFINE THE STATE
# ============================================  

class AgentState(TypedDict):
    """State definition for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    context: str
    response: str
    needs_retrieval: bool

# ============================================
# 2. CUSTOM CAPGEMINI EMBEDDINGS
# ============================================

CACHE_DIR = "embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


class CapgeminiEmbeddings(Embeddings):    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = , 
        model: str = "amazon.titan-embed-text-v2:0"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
                
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.base_url}/embeddings"
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result:
                    return [item["embedding"] for item in result["data"]]
                elif "embeddings" in result:
                    return result["embeddings"]
                else:
                    return result
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"Failed with endpoint {endpoint}: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._get_embedding(batch)
            all_embeddings.extend(embeddings)
            print(f"Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_embedding([text])[0]


# ============================================
# 3. PDF PROCESSOR
# ============================================

class PDFProcessor:
    """Handles PDF loading, text splitting, and vector store creation."""
    
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.documents = []
        
    def load_and_process(self) -> FAISS:
        """Load PDF, split into chunks, and create vector store."""
        
        print(f"Loading PDF from: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        raw_documents = loader.load()
        print(f"Loaded {len(raw_documents)} pages")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = text_splitter.split_documents(raw_documents)
        print(f"Split into {len(self.documents)} chunks")
        
        # Use Capgemini embeddings
        print("Initializing Capgemini embeddings...")
        embeddings = CapgeminiEmbeddings(
            api_key=API_KEY,
            base_url=OPENAI_COMPAT_URL,
            model=EMBEDDING_MODEL
        )
        
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(self.documents, embeddings)
        print("Vector store created successfully")
        
        return self.vector_store
    
    def get_retriever(self, k: int = 4):
        """Get a retriever from the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_and_process() first.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# ============================================
# 4. NODE FUNCTIONS
# ============================================

def create_nodes(retriever, llm):
    """Create all node functions for the graph."""
    
    def query_analyzer(state: AgentState) -> AgentState:
        """Analyze the user query to determine if document retrieval is needed."""
        query = state["query"]
        
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
        
        if query.lower().strip() in greetings:
            return {
                **state,
                "needs_retrieval": False,
                "context": ""
            }
        
        return {
            **state,
            "needs_retrieval": True
        }
    
    def retriever_node(state: AgentState) -> AgentState:
        """Retrieve relevant document chunks based on the user query."""
        if not state.get("needs_retrieval", True):
            return state
            
        query = state["query"]
        docs = retriever.invoke(query)
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            page_num = doc.metadata.get("page", "Unknown")
            context_parts.append(f"[Section {i} - Page {page_num}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            **state,
            "context": context
        }
    
    def response_generator(state: AgentState) -> AgentState:
        """Generate a response based on the context and user query."""
        query = state["query"]
        context = state.get("context", "")
        messages = state.get("messages", [])
        needs_retrieval = state.get("needs_retrieval", True)
        
        if needs_retrieval and context:
            system_prompt = ""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}")
            ])
        else:
            system_prompt = """You are a helpful assistant specialized in Medical Device Software Lifecycle
            Risk Management procedures.
            You can answer general questions and greetings.
            ANSWER QUESTION ONLY FROM THE PDF.  
            For specific questions about the document, 
            please ask the user to provide more details."""
                        
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}")
            ])
        
        history = messages[-10:] if len(messages) > 10 else messages
        
        chain = prompt | llm
        
        if needs_retrieval and context:
            response = chain.invoke({
                "context": context,
                "history": history,
                "query": query
            })
        else:
            response = chain.invoke({
                "history": history,
                "query": query
            })
        
        return {
            **state,
            "response": response.content,
            "messages": [HumanMessage(content=query), AIMessage(content=response.content)]
        }
    
    return query_analyzer, retriever_node, response_generator


# ============================================
# 5. CONDITIONAL EDGE FUNCTION
# ============================================

def should_retrieve(state: AgentState) -> str:
    """Determine if retrieval is needed based on the analyzed query."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "generate"


# ============================================
# 6. BUILD THE GRAPH
# ============================================

def build_graph(retriever, llm) -> StateGraph:
    """Build the LangGraph StateGraph with nodes and edges."""
    
    query_analyzer, retriever_node, response_generator = create_nodes(retriever, llm)
    
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_query", query_analyzer)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("generate", response_generator)
    

    workflow.set_entry_point("analyze_query")
    workflow.add_conditional_edges(
        "analyze_query",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compile
    app = workflow.compile()
    
    return app


# ============================================
# 7. MAIN AGENT CLASS
# ============================================

class PDFQAAgent:    
    def __init__(self, pdf_path: str, api_key: str = None):        
        self.api_key = api_key or API_KEY
        
        if not self.api_key:
            raise ValueError("API_KEY is required. Set it in environment or pass to constructor.")
        
       
        self.llm = ChatOpenAI(
            model=COMPLETION_MODEL,
            temperature=0.3,
            openai_api_key=self.api_key,
            openai_api_base=OPENAI_COMPAT_URL,
        )
        
        self.pdf_processor = PDFProcessor(pdf_path)
        self.vector_store = self.pdf_processor.load_and_process()
        self.retriever = self.pdf_processor.get_retriever(k=4)
        
        self.app = build_graph(self.retriever, self.llm)
        self.messages: List[BaseMessage] = []
    
    def ask(self, question: str) -> str:
        """Ask a question about the PDF document."""
        
        initial_state = {
            "messages": self.messages,
            "query": question,
            "context": "",
            "response": "",
            "needs_retrieval": True
        }
        
        result = self.app.invoke(initial_state)
        
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=result["response"]))
        
        return result["response"]
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_graph_visualization(self):
        """Get a visual representation of the graph."""
        return self.app.get_graph().draw_mermaid()


# ============================================
# 8. MAIN FUNCTION
# ============================================

def main():    
    pdf_path = ""
    if not API_KEY:
        print("Please set your API_KEY environment variable")
        return
    
    try:
        print("="*50)
        print("PDF Q&A Agent - Capgemini Generative Engine")
        print("="*50)
        print(f"LLM Model: {COMPLETION_MODEL}")
        print("="*50)
        
        print("\nInitializing PDF Q&A Agent...")
        agent = PDFQAAgent(pdf_path)
        
        print("\n" + "="*50)
        print("Agent ready! Ask questions about the document.")
        print("Commands: 'quit' to exit, 'clear' to clear history")
        print("="*50 + "\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            
            if question.lower() == 'clear':
                agent.clear_history()
                print("Conversation history cleared.\n")
                continue
            
            if not question:
                continue
            
            print("\nProcessing...")
            response = agent.ask(question)
            print(f"\nAssistant: {response}\n")
            
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
