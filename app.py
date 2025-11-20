import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

# Reranking Libraries
from sentence_transformers import CrossEncoder

# --- 1. Configuration & Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
LLM_MODEL = "llama3-8b-8192" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # The reranking model

st.set_page_config(layout="wide", page_title="Agentic RAG Chatbot")
st.title("🧠 Agentic RAG Chatbot (Self-Referential)")
st.caption("Architecture includes Agentic Router and Cross-Encoder Reranking.")

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
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name=LLM_MODEL
    )

@st.cache_resource
def get_reranker():
    """Load the cross-encoder reranking model."""
    return CrossEncoder(RERANKER_MODEL)

@st.cache_resource(show_spinner=False)
def setup_vector_store():
    """Load all project files from the deployed directory into FAISS."""
    with st.status("Initializing Knowledge Base...", expanded=True) as status:
        
        status.update(label="1. Loading project files (.py, .md, etc.)...", state="running")
        # Use DirectoryLoader to load all relevant files in the root directory
        loader = DirectoryLoader(
            ".", # Loads from the root of the Streamlit app (your GitHub repo)
            glob="**/*", # Recurse into all directories
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            # Exclude known files/folders that don't hold documentation
            exclude=["**/venv/*", "**/.git/*", "**/__pycache__/*", "*.lock", "*.txt", "**/streamlit/*"]
        )
        documents = loader.load()
        
        status.update(label=f"2. Splitting {len(documents)} files into chunks...", state="running")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
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
    if not router_llm: return 'COMPLEX' # Default to RAG if no LLM
    
    router_prompt = PromptTemplate.from_template(
        "You are an expert router. Classify the user's question into one of two categories: 'SIMPLE' (greetings, general chat, small talk) or 'COMPLEX' (questions requiring retrieval from the knowledge base, technical queries, or system-related questions). Respond with only the word 'SIMPLE' or 'COMPLEX'.\n\nQuestion: {question}"
    )
    
    # Use a faster, non-streaming call for the classification
    decision = router_llm.invoke(router_prompt.format(question=prompt)).content.strip().upper()
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
    """Handles COMPLEX RAG queries with Reranking."""
    llm = get_llm()
    reranker = get_reranker()
    
    # 1. Retrieval (Initial FAISS Search)
    with st.spinner("1. Searching vector database (FAISS retrieval)..."):
        retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Get 10 docs for reranking
        initial_docs = retriever.invoke(prompt)

    # 2. Reranking (Cross-Encoder)
    with st.spinner("2. Reranking top documents for maximum relevance..."):
        sentence_pairs = [[prompt, doc.page_content] for doc in initial_docs]
        # Get scores for all pairs
        scores = reranker.predict(sentence_pairs)
        
        # Pair documents with their scores and sort descending
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        
        # Select the top 4 documents for the final LLM context
        top_k_docs = [doc for score, doc in scored_docs[:4]]
        context = "\n\n".join([doc.page_content for doc in top_k_docs])
    
    # 3. Generation (LLM Synthesis)
    rag_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an expert RAG assistant built to answer questions ONLY about the project code and documentation provided in the context. Use ONLY the context. If the answer is not found, clearly state: 'I could not find the answer in the provided documents.'"
        ),
        HumanMessage(content="Context:\n---\n{context}\n---\nQuestion: {question}")
    ])
    
    chain = rag_prompt_template | llm
    
    return chain.stream({"context": context, "question": prompt})


# --- 5. Streamlit UI & Orchestration ---

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = setup_vector_store()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.caption("Your entire project source code is loaded as the knowledge base for this chatbot.")
    st.caption(f"LLM: {LLM_MODEL} (Groq)")
    st.caption(f"Reranker: {RERANKER_MODEL}")

# Main Chat Interface
if st.session_state.vector_store is None:
    st.error("Knowledge base failed to initialize. Check if the required files and dependencies are available.")
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
            
            st.info(f"Router Decision: **{router_decision}**. Processing...")
            
            # 2. Execute based on Decision
            placeholder = st.empty()
            full_response = ""
            
            if router_decision == 'COMPLEX':
                response_stream = get_rag_response(st.session_state.vector_store, prompt)
            else: # SIMPLE
                with st.spinner("3. Generating simple response..."):
                    response_stream = get_simple_response(prompt)

            # 3. Stream output
            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌") 
            
            placeholder.markdown(full_response)
            
        # Store final response in session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})