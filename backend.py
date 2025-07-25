# backend.py

import pickle
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from rag_pipeline import create_rag_chain, load_llm

app = FastAPI(title="DocuMentor AI Backend", version="1.0.0")

llm = load_llm()
vector_stores: Dict[str, FAISS] = {}
qa_chains: Dict[str, Optional[object]] = {}

class QueryRequest(BaseModel):
    session_id: str
    query: str

class ProcessRequest(BaseModel):
    session_id: str
    docs_pickle: str

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "DocuMentor AI Backend"}

@app.post("/process_docs")
def process_documents(request: ProcessRequest):
    """Process documents and create vector store for session."""
    try:
        docs_bytes = bytes.fromhex(request.docs_pickle)
        documents = pickle.loads(docs_bytes)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
        
        vector_stores[request.session_id] = vector_store
        qa_chains[request.session_id] = create_rag_chain(llm, vector_store)
        
        return {
            "status": "success",
            "message": f"Processed {len(documents)} document chunks"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: QueryRequest):
    """Process chat query and return response."""
    qa_chain = qa_chains.get(request.session_id)
    
    if not qa_chain:
        raise HTTPException(
            status_code=400,
            detail="No documents processed for this session. Please upload documents first."
        )
    
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"answer": response.get("result", "I couldn't find a relevant answer.")}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/status")
def get_session_status(session_id: str):
    """Get status of a session."""
    has_documents = session_id in vector_stores
    return {
        "session_id": session_id,
        "has_documents": has_documents,
        "ready": session_id in qa_chains and qa_chains[session_id] is not None
    }