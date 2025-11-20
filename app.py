import streamlit as st
import os
import time

# General LangChain/LLM Components
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# RAG/Data Loading Components
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reranking Libraries
from sentence_transformers import CrossEncoder

# --- 1. Configuration & Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
LLM_MODEL = "llama3-8b-8192" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

st.set_page_config(layout="wide", page_title="Agentic RAG Chatbot")
st.title("🧠 Agentic RAG Chatbot (Self-Referential)")
st.caption("Now includes Agentic Router, Cross-Encoder Reranking, and Grounding Guard Rails with Citations.")

# --- 2. Resource Caching for RAG Components ---

@st.cache_resource
def get_embeddings():
    """Load the CPU-optimized embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    """Initialize the Groq LLM Client."""
    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
        return None
    return ChatGroq(
        temperature=0, # Use low temperature for deterministic RAG answers
        groq_api_key=GROQ_API_KEY, 
        model_name=LLM_MODEL
    )

@st.cache_resource
def get_reranker():
    """Load the cross-encoder reranking model."""
    return CrossEncoder(RERANKER_MODEL)

@st.cache_resource(show_spinner=False)
def setup_vector_store():
    """Load all project files from the deployed directory into FAISS using a loader map."""
    with st.status("Initializing Knowledge Base...", expanded=True) as status:
        
        status.update(label="1. Loading project files (.py, .md, .pdf, etc.)...", state="running")
        
        # DirectoryLoader with a map ensures different file types are handled correctly
        loader = DirectoryLoader(
            ".", # Loads from the root of the Streamlit app (your GitHub repo)
            glob="**/*", # Recurse into all directories
            loader_map={
                ".txt": (TextLoader, {"autodetect_encoding": True}),
                ".md": (TextLoader, {"autodetect_encoding": True}),
                ".py": (TextLoader, {"autodetect_encoding": True}),
                ".pdf": (PyPDFLoader, {}), 
            },
            recursive=True,
            # Exclude common dev folders and lock files
            exclude=["**/venv/*", "**/.git/*", "**/__pycache__/*", "*.lock", "*.txt", "**/streamlit/*"]
        )
        documents = loader.load()
        
        status.update(label=f"2. Splitting {len(documents)} files into chunks...", state="running")
        text_splitter = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        status.update(label="3. Generating embeddings and building FAISS index...", state="running")
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        status.update(label=f"Knowledge Base Ready! {len(chunks)} chunks loaded.", state="complete", expanded=False)
        return vector_store

# --- 3. Agentic Router Logic ---

def get_router_decision(prompt):
    """Uses the LLM to decide if the query is simple or needs RAG (Agentic Router)."""
    router_llm = get_llm()
    if not router_llm: return 'COMPLEX' 
    
    router_prompt = PromptTemplate.from_template(
        "You are an expert router. Classify the user's question into one of two categories: 'SIMPLE' (greetings, general chat, small talk, or non-technical topics) or 'COMPLEX' (questions requiring retrieval from the knowledge base, technical queries, or system-related questions). Respond with only the word 'SIMPLE' or 'COMPLEX'.\n\nQuestion: {question}"
    )
    
    decision = router_llm.invoke(router_prompt.format(question=prompt)).content.strip().upper()
    
    # Simple validation, default to RAG for safety
    if decision not in ['SIMPLE', 'COMPLEX']:
        return 'COMPLEX'
    return decision

# --- 4. RAG and Simple Response Handlers ---

def get_simple_response(prompt):
    """Handles SIMPLE chit-chat queries."""
    llm = get_llm()
    simple_prompt = PromptTemplate.from_template(
        "You are a friendly, witty, and helpful general assistant. Respond concisely to this question: {question}"
    )
    return llm.stream(simple_prompt.format(question=prompt))

def get_rag_response(vector_store, prompt):
    """Handles COMPLEX RAG queries with Reranking and implements Guard Rail/Metadata."""
    llm = get_llm()
    reranker = get_reranker()
    
    # 1. Retrieval (Initial FAISS Search)
    with st.spinner("1. Searching vector database (FAISS retrieval)..."):
        retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Get 10 docs for reranking
        initial_docs = retriever.invoke(prompt)

    # 2. Reranking (Cross-Encoder)
    with st.spinner("2. Reranking top documents for maximum relevance..."):
        sentence_pairs = [[prompt, doc.page_content] for doc in initial_docs]
        scores = reranker.predict(sentence_pairs)
        
        # Pair documents with their scores and sort descending
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        
        # Select the top 4 documents for the final LLM context
        top_k_docs = [doc for score, doc in scored_docs[:4]]
        context = "\n\n".join([doc.page_content for doc in top_k_docs])

        # Extract unique sources for citation (Metadata Implementation)
        unique_sources = sorted(list(set(doc.metadata.get('source', 'Unknown File') for doc in top_k_docs)))
        citation_text = "\n\n---\n**Sources Used (Metadata):**\n" + "\n".join([f"- `{source}`" for source in unique_sources])
    
    # 3. Generation (LLM Synthesis with Guard Rail Prompt)
    rag_prompt_template = ChatPromptTemplate.from_messages([
        # The System Message acts as the primary Guard Rail
        SystemMessage(
            "You are an expert RAG assistant built to answer questions ONLY about the project code and documentation provided in the context. "
            "Your output MUST be entirely based on the 'Context' section below. "
            "If the provided documents do not contain the answer, you MUST clearly and politely state: 'I could not find the answer in the provided project documents.'"
        ),
        HumanMessage(content="Context:\n---\n{context}\n---\nQuestion: {question}")
    ])
    
    chain = rag_prompt_template | llm
    
    return chain.stream({"context": context, "question": prompt}), citation_text


# --- 5. Streamlit UI & Orchestration ---

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = setup_vector_store()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("## RAG Architecture Details")
    st.markdown("""
    This chatbot operates using a decoupled, agentic architecture:
    1.  **Data Ingestion (Metadata):** Scans the project directory (including `.py`, `.md`, `.pdf`) and extracts file paths as metadata.
    2.  **Agentic Router:** Uses the LLM to classify the query as `SIMPLE` (chit-chat) or `COMPLEX` (RAG required).
    3.  **RAG Pipeline:** For `COMPLEX` queries:
        * Retrieves 10 documents from FAISS.
        * **Cross-Encoder Reranker** re-sorts them for relevance.
        * **Guard Rail:** A strict system prompt ensures the LLM stays grounded to the context.
        * **Citation:** The source file paths (metadata) of the documents used are appended to the response.
    """)
    st.caption(f"LLM: {LLM_MODEL} (Groq)")
    st.caption(f"Reranker: {RERANKER_MODEL}")

# Main Chat Interface
if st.session_state.vector_store is None:
    st.error("Knowledge base failed to initialize. Please check API key and dependencies.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the project code or architecture..."):
        # 1. Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            
            # --- AGENTIC ROUTER EXECUTION ---
            with st.spinner("Routing query..."):
                router_decision = get_router_decision(prompt)
            
            st.info(f"Router Decision: **{router_decision}**.")
            
            # 2. Execute based on Decision
            placeholder = st.empty()
            full_response = ""
            
            if router_decision == 'COMPLEX':
                response_stream, citation_text = get_rag_response(st.session_state.vector_store, prompt)
            else: # SIMPLE
                with st.spinner("3. Generating simple response..."):
                    response_stream = get_simple_response(prompt)
                citation_text = "" # No sources for simple chat

            # 3. Stream output
            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌") 
            
            # Append citation text and finalize the output
            full_response += citation_text
            placeholder.markdown(full_response)
            
        # Store final response in session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})