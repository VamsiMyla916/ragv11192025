import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import time # Used for simulated agentic steps

# --- 1. Configuration & Setup ---

# The environment variable Streamlit Cloud will use.
# NOTE: GROQ_API_KEY must be set in Streamlit Secrets!
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Groq Model (Llama 3 8B is excellent and fast)
LLM_MODEL = "llama3-8b-8192" 
# Embedding Model (Optimized for CPU/Serverless)
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

st.set_page_config(layout="wide", page_title="Free Serverless RAG Chatbot")
st.title("📄 Free, Ultra-Fast RAG Chatbot (Powered by Groq)")

# --- 2. Resource Caching for RAG Components ---

@st.cache_resource
def get_embeddings():
    """Load the sentence-transformer model (CPU-optimized)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    """Initialize the Groq LLM Client (external API)."""
    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY not found. Please set it in Streamlit Secrets.")
        return None
    
    return ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name=LLM_MODEL
    )

@st.cache_resource(show_spinner=False)
def setup_vector_store(pdf_data):
    """Processes PDF data to create a FAISS vector store in memory."""
    with st.status("Processing Document...", expanded=True) as status:
        # Save PDF data to a temporary file
        temp_file_path = "temp_doc.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(pdf_data)
        
        status.update(label="Loading document with PyPDFLoader...", state="running")
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        status.update(label="Splitting document into chunks...", state="running")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        status.update(label="Generating embeddings and building FAISS index...", state="running")
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Clean up the temporary file (important for serverless envs)
        os.remove(temp_file_path)
        
        status.update(label=f"Document processed successfully! {len(chunks)} chunks loaded.", state="complete", expanded=False)
        return vector_store

# --- 3. RAG Core Logic ---

def get_rag_response(vector_store, prompt):
    """Handles vector retrieval and LLM generation (The Agentic RAG Step)."""
    llm = get_llm()
    if not llm:
        return "LLM service initialization failed.", None
    
    # Simulate agentic steps (previously handled by FastAPI streaming)
    with st.spinner("1. Searching vector database..."):
        time.sleep(0.5)
        # 1. Retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(prompt)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    with st.spinner("2. Preparing prompt and calling Groq LLM..."):
        # 2. Prompt Template (Grounding the response)
        rag_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(
                f"You are a helpful and ultra-fast RAG assistant. Use ONLY the provided context to answer the question. If the answer is not found in the context, clearly state: 'I could not find the answer in the provided documents.' Model used: {LLM_MODEL}"
            ),
            HumanMessage(content="Context:\n---\n{context}\n---\nQuestion: {question}")
        ])
        
        # 3. Create the Chain
        chain = rag_prompt_template | llm
        
        # 4. Invoke the Chain and Stream the Response
        full_response = ""
        placeholder = st.empty()
        
        # Use LangChain's stream method for real-time output
        for chunk in chain.stream({"context": context, "question": prompt}):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌") # Use cursor effect
        
        placeholder.markdown(full_response)
        return full_response, retrieved_docs


# --- 4. Streamlit UI & Session Management ---

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF Upload
with st.sidebar:
    st.header("1. Upload Knowledge Source (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.session_state.vector_store is None:
        st.session_state.pdf_name = uploaded_file.name
        pdf_bytes = uploaded_file.read()
        st.session_state.vector_store = setup_vector_store(pdf_bytes)
        # Clear messages after a new file is uploaded
        st.session_state.messages = []
    
    st.markdown("---")
    st.caption("Deployment Architecture: Streamlit Cloud (UI/RAG Retrieval) + Groq (Ultra-Fast LLM Inference)")

# Main Chat Interface
if st.session_state.vector_store is None:
    st.info("Upload a PDF in the sidebar to start the RAG chatbot.")
else:
    st.markdown(f"**Document Loaded:** {st.session_state.pdf_name}")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream assistant response
        with st.chat_message("assistant"):
            response_text, retrieved_docs = get_rag_response(st.session_state.vector_store, prompt)
            
        # Store final response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})