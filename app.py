import os
import requests
import streamlit as st
from typing import List
from urllib.parse import quote
from st_supabase_connection import SupabaseConnection

st.set_page_config(
    page_title="DocuMentor AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    conn = st.connection("supabase", type=SupabaseConnection)
except Exception as e:
    st.error("Could not connect to Supabase. Please ensure secrets are configured correctly.")
    st.stop()


BACKEND_URL = os.getenv("BACKEND_URL", "https://documentor-ai-production.up.railway.app")

def get_auth_headers():
    """Constructs authorization headers from the stored access token"""
    if 'user' not in st.session_state or st.session_state.user is None:
        return None
    
    session = conn.auth.get_session()
    if not session or not session.access_token:
        st.session_state.clear()
        st.rerun()
        return None
        
    return {"Authorization": f"Bearer {session.access_token}"}

def process_documents(uploaded_files: List) -> dict:
    headers = get_auth_headers()
    if not headers:
        return {"status": "error", "message": "User not logged in."}
    files_payload = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
    try:
        response = requests.post(f"{BACKEND_URL}/process_docs", files=files_payload, headers=headers, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"status": "error", "message": f"Server error: {http_err.response.text}"}
    except requests.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

def send_query(query: str) -> dict:
    headers = get_auth_headers()
    if not headers:
        return {"error": "User not logged in."}
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": query, "model_name": st.session_state.get("model_name", "gemini-2.5-flash")},
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}

def get_user_documents():
    headers = get_auth_headers()
    if not headers:
        return []
    try:
        response = requests.get(f"{BACKEND_URL}/documents", headers=headers)
        if response.status_code == 200:
            return response.json().get("files", [])
    except requests.RequestException:
        return []
    return []

def delete_document(doc_name: str):
    headers = get_auth_headers()
    if not headers:
        return {"status": "error", "message": "User not logged in."}
    try:
        encoded_doc_name = quote(doc_name)
        response = requests.delete(f"{BACKEND_URL}/documents/{encoded_doc_name}", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"status": "error", "message": f"Server error: {http_err.response.text}"}
    except requests.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

def get_chat_history():
    headers = get_auth_headers()
    if not headers:
        return []
    try:
        response = requests.get(f"{BACKEND_URL}/sessions", headers=headers)
        if response.status_code == 200:
            return response.json().get("messages", [])
    except requests.RequestException:
        return []
    return []

def save_chat_history():
    headers = get_auth_headers()
    if not headers:
        return
    try:
        requests.post(
            f"{BACKEND_URL}/sessions",
            json={"messages": st.session_state.messages},
            headers=headers
        )
    except requests.RequestException:
        pass

def delete_chat_history():
    """Sends a request to the backend to delete the user's chat history"""
    headers = get_auth_headers()
    if not headers:
        return {"status": "error", "message": "User not logged in."}
    try:
        response = requests.delete(f"{BACKEND_URL}/sessions", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"status": "error", "message": f"Server error: {http_err.response.text}"}
    except requests.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}


if 'user' not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("Welcome to DocuMentor AI")
    st.subheader("Please log in or create an account to continue.")

    login_tab, register_tab, reset_tab = st.tabs(["Login", "Register", "Forgot Password"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        session = conn.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user = session.user
                        st.rerun()
                    except Exception as e:
                        st.error("Login failed. Please check your credentials or verify your email.")

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            register_button = st.form_submit_button("Register")

            if register_button:
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    try:
                        response = conn.auth.sign_up({"email": email, "password": password})
                        if response.user and not response.user.identities:
                            st.error("This email is already registered. Please use the Login tab.")
                        else:
                            st.success("Confirmation email sent! Please check your inbox to verify your account.")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")

    with reset_tab:
        with st.form("reset_form"):
            email = st.text_input("Email", key="reset_email")
            reset_button = st.form_submit_button("Send Password Reset Link")

            if reset_button:
                if not email:
                    st.error("Please enter your email address.")
                else:
                    try:
                        conn.auth.reset_password_for_email(email)
                        st.success("Password reset link sent! Please check your email.")
                    except Exception as e:
                        st.error(f"Could not send reset link: {e}")

else:
    with st.sidebar:
        st.title("Welcome!")
        st.write(f"Logged in as: **{st.session_state.user.email}**")
        if st.button("Logout"):
            conn.auth.sign_out()
            st.session_state.clear()
            st.rerun()
        
        if st.button("Clear Chat History", type="secondary"):
            with st.spinner("Clearing history..."):
                result = delete_chat_history()
                if result.get("status") == "success":
                    if "messages" in st.session_state:
                        del st.session_state["messages"]
                    st.rerun()
                else:
                    st.error("Could not clear history.")

        st.divider()

        st.header("⚙️ Model Selection")
        model_options = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        st.session_state.model_name = st.selectbox(
            "Choose a Gemini model:", options=model_options,
            index=model_options.index(st.session_state.get("model_name", "gemini-2.5-flash"))
        )

        st.divider()

        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader("Select PDF files to analyze", accept_multiple_files=True, type="pdf")
        if st.button("Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing your documents..."):
                    result = process_documents(uploaded_files)
                    if result.get("status") == "success":
                        st.success("Documents processed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
            else:
                st.warning("Please select at least one PDF file.")
        
        st.divider()
        
        with st.expander("🗂️ Your Documents", expanded=True):
            with st.spinner("Loading document list..."):
                user_files = get_user_documents()
                if user_files:
                    for doc_name in user_files:
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            st.info(doc_name, icon="📄")
                        with col2:
                            if st.button("🗑️", key=f"delete_{doc_name}", help=f"Delete {doc_name}"):
                                with st.spinner(f"Deleting {doc_name}..."):
                                    result = delete_document(doc_name)
                                    if result.get("status") == "success":
                                        if "messages" in st.session_state:
                                            del st.session_state["messages"]
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete: {result.get('message')}")
                else:
                    st.info("No documents uploaded yet.")

    st.title("🤖 DocuMentor AI")
    st.caption(f"Chat with your documents using **{st.session_state.get('model_name')}**")

    if "messages" not in st.session_state:
        st.session_state.messages = get_chat_history()
        if not st.session_state.messages:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload some documents to get started."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing with {st.session_state.get('model_name')}..."):
                response = send_query(prompt)
                answer = response.get("answer", f"Sorry, an error occurred: {response.get('error')}")
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_chat_history()
