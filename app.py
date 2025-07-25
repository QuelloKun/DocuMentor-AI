# app.py

import os
import pickle
import tempfile
import uuid
import json
from typing import List
from datetime import datetime

import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_DIR = "/app/data/sessions"

st.set_page_config(
    page_title="DocuMentor AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def ensure_data_dir():
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)

def save_session_data(custom_name=None):
    """Save current session data to file."""
    if "session_id" not in st.session_state:
        return
    
    session_file = os.path.join(DATA_DIR, f"{st.session_state.session_id}.json")
    session_data = {
        "session_id": st.session_state.session_id,
        "name": custom_name or st.session_state.get("session_name", "Untitled Session"),
        "messages": st.session_state.get("messages", []),
        "docs_processed": st.session_state.get("docs_processed", False),
        "last_updated": datetime.now().isoformat()
    }
    
    try:
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save session: {e}")

def load_session_data(session_id: str) -> dict:
    """Load session data from file."""
    session_file = os.path.join(DATA_DIR, f"{session_id}.json")
    
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load session: {e}")
    
    return {}

def get_available_sessions() -> List[dict]:
    """Get list of available sessions."""
    ensure_data_dir()
    sessions = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            session_data = load_session_data(session_id)
            if session_data:
                sessions.append({
                    "id": session_id,
                    "name": session_data.get("name", "Untitled Session"),
                    "last_updated": session_data.get("last_updated", "Unknown"),
                    "message_count": len(session_data.get("messages", [])),
                    "docs_processed": session_data.get("docs_processed", False)
                })
    
    # Sort by last updated (most recent first)
    sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    return sessions

def delete_session(session_id: str):
    """Delete a session file."""
    session_file = os.path.join(DATA_DIR, f"{session_id}.json")
    try:
        if os.path.exists(session_file):
            os.remove(session_file)
            return True
    except Exception as e:
        st.error(f"Failed to delete session: {e}")
    return False

def rename_session(session_id: str, new_name: str):
    """Rename a session."""
    session_data = load_session_data(session_id)
    if session_data:
        session_data["name"] = new_name
        session_data["last_updated"] = datetime.now().isoformat()
        session_file = os.path.join(DATA_DIR, f"{session_id}.json")
        try:
            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to rename session: {e}")
    return False

def initialize_session_state():
    """Initialize session state variables."""
    ensure_data_dir()
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_data = load_session_data(st.session_state.session_id)
    
    if "messages" not in st.session_state:
        st.session_state.messages = session_data.get("messages", [
            {"role": "assistant", "content": "Hello! Upload some PDFs and I'll help you analyze them."}
        ])
    
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = session_data.get("docs_processed", False)

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
        # Session Management
        st.header("💾 Session Management")
        
        available_sessions = get_available_sessions()
        if available_sessions:
            st.write("**Recent Sessions:**")
            for session in available_sessions[:5]:  # Show last 5 sessions
                with st.expander(f"{session['name']} ({session['message_count']} msgs)" + (" 📄" if session['docs_processed'] else "")):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if st.button("Load", key=f"load_{session['id']}", help="Load this session"):
                            st.session_state.session_id = session['id']
                            session_data = load_session_data(session['id'])
                            st.session_state.messages = session_data.get("messages", [])
                            st.session_state.docs_processed = session_data.get("docs_processed", False)
                            st.session_state.session_name = session_data.get("name", "Untitled Session")
                            st.rerun()
                    
                    with col2:
                        if st.button("✏️", key=f"rename_{session['id']}", help="Rename session"):
                            st.session_state[f"renaming_{session['id']}"] = True
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️", key=f"delete_{session['id']}", help="Delete session"):
                            if delete_session(session['id']):
                                st.success("Session deleted!")
                                st.rerun()
                    
                    # Rename input field
                    if st.session_state.get(f"renaming_{session['id']}", False):
                        new_name = st.text_input("New name:", value=session['name'], key=f"new_name_{session['id']}")
                        col_save, col_cancel = st.columns([1, 1])
                        with col_save:
                            if st.button("Save", key=f"save_name_{session['id']}"):
                                if rename_session(session['id'], new_name):
                                    st.success("Session renamed!")
                                    st.session_state[f"renaming_{session['id']}"] = False
                                    st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_name_{session['id']}"):
                                st.session_state[f"renaming_{session['id']}"] = False
                                st.rerun()
        
        # Current session name
        current_session_name = st.session_state.get("session_name", "Untitled Session")
        st.write(f"**Current:** {current_session_name}")
        
        col_new, col_rename = st.columns([2, 1])
        with col_new:
            if st.button("🆕 New Session", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Initialize new session
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_name = "Untitled Session"
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! Upload some PDFs and I'll help you analyze them."}
                ]
                st.session_state.docs_processed = False
                
                # Save the new session immediately
                save_session_data()
                
                # Force a rerun to refresh the UI
                st.rerun()
        
        with col_rename:
            if st.button("✏️", help="Rename current session"):
                st.session_state["renaming_current"] = True
                st.rerun()
        
        # Rename current session
        if st.session_state.get("renaming_current", False):
            new_name = st.text_input("Rename current session:", value=current_session_name, key="rename_current_input")
            col_save, col_cancel = st.columns([1, 1])
            with col_save:
                if st.button("Save", key="save_current_name"):
                    st.session_state.session_name = new_name
                    save_session_data(new_name)
                    st.session_state["renaming_current"] = False
                    st.success("Session renamed!")
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="cancel_current_name"):
                    st.session_state["renaming_current"] = False
                    st.rerun()
        
        st.divider()
        
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
                        save_session_data()
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
            save_session_data()

def main():
    """Main application function."""
    initialize_session_state()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()