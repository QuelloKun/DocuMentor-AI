# app.py

import os
import pickle
import tempfile
import uuid
from typing import List

import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BACKEND_URL = "http://backend:8000"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

st.set_page_config(
    page_title="DocuMentor AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Upload some PDFs and I'll help you analyze them."}
        ]
    
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False

def process_documents(uploaded_files: List) -> dict:
    """Process uploaded PDF files and send to backend."""
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(documents)
    
    docs_pickle = pickle.dumps(splits)
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/process_docs",
            json={
                "session_id": st.session_state.session_id,
                "docs_pickle": docs_pickle.hex()
            },
            timeout=30
        )
        return response.json()
    except requests.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

def send_query(query: str) -> dict:
    """Send query to backend and get response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "query": query
            },
            timeout=30
        )
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def render_sidebar():
    """Render the document upload sidebar."""
    with st.sidebar:
        st.header("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Select PDF files to analyze",
            accept_multiple_files=True,
            type="pdf",
            help="Upload one or more PDF files to get started"
        )
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing your documents..."):
                    result = process_documents(uploaded_files)
                    
                    if result.get("status") == "success":
                        st.success("Documents processed successfully!")
                        st.session_state.docs_processed = True
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
            else:
                st.warning("Please select at least one PDF file.")
        
        if st.session_state.docs_processed:
            st.success("✅ Documents ready for questions")

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 DocuMentor AI")
    st.caption("Chat with Your Research Papers")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if st.session_state.docs_processed:
                with st.spinner("Analyzing your documents..."):
                    response = send_query(prompt)
                    
                    if "answer" in response:
                        answer = response["answer"]
                    elif "error" in response:
                        answer = f"Sorry, I encountered an error: {response['error']}"
                    else:
                        answer = "Sorry, I couldn't process your question."
                    
                    st.markdown(answer)
            else:
                answer = "Please upload and process your documents first."
                st.warning(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

def main():
    """Main application function."""
    initialize_session_state()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()