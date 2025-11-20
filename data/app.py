import streamlit as st
import os

# General LangChain/LLM Components
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# RAG/Data Loading Components
from langchain_community.document_loaders import DirectoryLoader
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

st.set_page_config(layout="wide", page_title="Re-Coupled Agentic RAG Chatbot")
st.title("🧠 Re-Coupled Agentic RAG Chatbot")
st.caption("All router, RAG, and reranking logic is now integrated for single-process cloud deployment.")

# --- 2. Resource Caching for RAG Components ---

@st.cache_resource
def get_embeddings():
    """Load the CPU-optimized embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    """Initialize the Groq LLM Client."""
    if not GROQ_API_KEY:
        st.error("ERROR: GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
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
    """Load project files into FAISS using a loader map (Metadata source implementation)."""
    with st.status("Initializing Knowledge Base (Loading Project Files)...", expanded=True) as status:

        status.update(label="1. Scanning project directory and mapping loaders...", state="running")
        
        loader = DirectoryLoader(
            ".",
            glob="**/*.{txt,md,py,pdf}",
            recursive=True,
            silent_errors=True,
            exclude=["**/venv/*", "**/.git/*", "**/__pycache__/*", "**/streamlit/*", "requirements.txt", "README.md", "*.lock"]
        )
        documents = loader.load()
        if not documents:
            st.error("No valid documents loaded! Please upload project files with .txt, .md, .py, or .pdf extension.")
            st.stop()

        status.update(label=f"2. Splitting {len(documents)} source documents into chunks...", state="running")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            st.error("No text chunks were produced from the documents. Are your files empty?")
            st.stop()
        
        status.update(label="3. Building FAISS index with embeddings...", state="running")
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        status.update(label=f"Knowledge Base Ready! {len(chunks)} chunks loaded.", state="complete", expanded=False)
        return vector_store

# --- 3. Agentic Router Implementation ---

def get_router_decision(prompt):
    """Uses the LLM to classify the query as SIMPLE or COMPLEX (RAG)."""
    router_llm = get_llm()
    if not router_llm:
        return 'COMPLEX' 
    router_prompt = PromptTemplate.from_template(
        "You are an expert router. Classify the user's question into one of two categories: 'SIMPLE' (greetings, general chat, non-technical topics) or 'COMPLEX' (questions requiring retrieval from the knowledge base, technical queries about the project, or system-related questions). Respond with only the word 'SIMPLE' or 'COMPLEX'.\n\nQuestion: {question}"
    )
    decision = router_llm.invoke(router_prompt.format(question=prompt)).content.strip().upper()
    return decision if decision in ['SIMPLE', 'COMPLEX'] else 'COMPLEX'

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
    
    # 1. Retrieval (FAISS Search)
    with st.spinner("1. Retrieving 10 candidate documents from vector database..."):
        retriever = vector_store.as_retriever(search_kwargs={"k": 10}) 
        initial_docs = retriever.invoke(prompt)

    # 2. Reranking (Cross-Encoder)
    with st.spinner("2. Reranking documents based on cross-encoder scores..."):
        sentence_pairs = [[prompt, doc.page_content] for doc in initial_docs]
        scores = reranker.predict(sentence_pairs)
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        top_k_docs = [doc for score, doc in scored_docs[:4]] # Use top 4 for context
        context = "\n\n".join([doc.page_content for doc in top_k_docs])
        unique_sources = sorted(list(set(doc.metadata.get('source', 'Unknown File') for doc in top_k_docs)))
        citation_text = "\n\n---\n**Sources Used (Metadata/Citations):**\n" + "\n".join([f"- `{source}`" for source in unique_sources])
    
    # 3. Generation (LLM Synthesis with Guard Rail Prompt)
    rag_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a project-specific RAG assistant. You MUST ONLY use the provided 'Context' below to answer the user's question. "
            "DO NOT use external or general knowledge. "
            "If the answer cannot be found in the provided documents, you MUST clearly and politely state: 'I could not find the answer in the provided project documents.'"
        ),
        HumanMessage(content="Context:\n---\n{context}\n---\nQuestion: {question}")
    ])
    chain = rag_prompt_template | llm
    return chain.stream({"context": context, "question": prompt}), citation_text

# --- 5. Streamlit UI & Orchestration ---

# Initialize State
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = setup_vector_store()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for architecture overview
with st.sidebar:
    st.markdown("## Architecture: Re-Coupled for Cloud Deployment")
    st.markdown("""
    This application integrates your **Agentic Router**, **Reranker**, and **RAG Core** into a single process for simplified deployment on Streamlit Cloud.
    - **Caching:** `@st.cache_resource` ensures fast vector store access.
    - **Guard Rail:** System prompt enforces grounded answers.
    - **Citations:** Source file paths are displayed as metadata.
    """)

if st.session_state.vector_store is None:
    st.error("Knowledge base failed to initialize. Please check your dependencies and API key.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the project..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing query (SIMPLE or COMPLEX)..."):
                router_decision = get_router_decision(prompt)
            st.info(f"Router Decision: **{router_decision}**.")
            placeholder = st.empty()
            full_response = ""
            citation_text = ""
            if router_decision == 'COMPLEX':
                response_stream, citation_text = get_rag_response(st.session_state.vector_store, prompt)
            else:
                with st.spinner("3. Generating simple chat response..."):
                    response_stream = get_simple_response(prompt)

            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")
            full_response += citation_text
            placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
