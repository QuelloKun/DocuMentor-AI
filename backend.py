import os
import jwt
import json
import boto3
import tempfile
from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
from botocore.exceptions import ClientError


app = FastAPI(title="DocuMentor AI Backend", version="1.5.0")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
ALGORITHM = "HS256"

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

class QueryRequest(BaseModel):
    query: str
    model_name: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatHistory(BaseModel):
    messages: List[ChatMessage]

async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    """Validates Supabase JWT and returns the user's unique ID (sub)"""
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT Secret not configured on backend.")
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=[ALGORITHM],
            audience="authenticated"
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token: User ID not found.")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Could not validate credentials: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/documents")
def get_user_documents(user_id: str = Depends(get_current_user_id)):
    document_s3_prefix = f"uploaded_pdfs/{user_id}/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=document_s3_prefix)
        files = [os.path.basename(item['Key']) for item in response.get('Contents', [])]
        return {"status": "success", "files": files}
    except Exception as e:
        return {"status": "error", "files": [], "detail": str(e)}

@app.delete("/documents/{doc_name}")
def delete_document(doc_name: str, user_id: str = Depends(get_current_user_id)):
    pdf_s3_key = f"uploaded_pdfs/{user_id}/{doc_name}"
    faiss_index_key = f"session_indexes/{user_id}/index.faiss"
    faiss_pkl_key = f"session_indexes/{user_id}/index.pkl"
    objects_to_delete = [{'Key': pdf_s3_key}, {'Key': faiss_index_key}, {'Key': faiss_pkl_key}]
    try:
        s3.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': objects_to_delete, 'Quiet': True})
        return {"status": "success", "message": f"Document '{doc_name}' and associated index deleted."}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 Error during deletion: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {e}")

@app.get("/sessions", response_model=ChatHistory)
def get_chat_history(user_id: str = Depends(get_current_user_id)):
    session_key = f"sessions/{user_id}.json"
    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=session_key)
        history_data = json.loads(response["Body"].read().decode("utf-8"))
        return ChatHistory(messages=history_data.get("messages", []))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return ChatHistory(messages=[])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
def save_chat_history(history: ChatHistory, user_id: str = Depends(get_current_user_id)):
    session_key = f"sessions/{user_id}.json"
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=session_key,
            Body=json.dumps(history.dict(), indent=2),
            ContentType="application/json",
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions")
def delete_chat_history(user_id: str = Depends(get_current_user_id)):
    """Deletes the chat history file for the logged-in user from S3"""
    session_key = f"sessions/{user_id}.json"
    try:
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=session_key)
        return {"status": "success", "message": "Chat history deleted."}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {"status": "success", "message": "No chat history to delete."}
        raise HTTPException(status_code=500, detail=f"S3 Error during deletion: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {e}")

@app.post("/process_docs")
async def process_documents(files: List[UploadFile] = File(...), user_id: str = Depends(get_current_user_id)):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            temp_pdf_path = os.path.join(temp_dir, file.filename)
            file_content = await file.read()
            with open(temp_pdf_path, "wb") as f:
                f.write(file_content)
            pdf_s3_key = f"uploaded_pdfs/{user_id}/{file.filename}"
            s3.put_object(Bucket=S3_BUCKET_NAME, Key=pdf_s3_key, Body=file_content)
            loader = PyPDFLoader(temp_pdf_path)
            documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        faiss_temp_path = os.path.join(temp_dir, "faiss_index")
        vector_store.save_local(faiss_temp_path)
        s3.upload_file(os.path.join(faiss_temp_path, "index.faiss"), S3_BUCKET_NAME, f"session_indexes/{user_id}/index.faiss")
        s3.upload_file(os.path.join(faiss_temp_path, "index.pkl"), S3_BUCKET_NAME, f"session_indexes/{user_id}/index.pkl")
    return {"status": "success", "message": f"Processed {len(splits)} chunks from {len(files)} files."}

@app.post("/chat")
def chat(request: QueryRequest, user_id: str = Depends(get_current_user_id)):
    from rag_pipeline import create_rag_chain, load_llm
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    index_s3_prefix = f"session_indexes/{user_id}"
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            faiss_local_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(faiss_local_path, exist_ok=True)
            s3.download_file(S3_BUCKET_NAME, f"{index_s3_prefix}/index.faiss", os.path.join(faiss_local_path, "index.faiss"))
            s3.download_file(S3_BUCKET_NAME, f"{index_s3_prefix}/index.pkl", os.path.join(faiss_local_path, "index.pkl"))
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(faiss_local_path, embeddings, allow_dangerous_deserialization=True)
            llm = load_llm(model_name=request.model_name)
            qa_chain = create_rag_chain(llm, vector_store)
            response = qa_chain.invoke({"query": request.query})
            return {"answer": response.get("result", "I couldn't find a relevant answer.")}
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Documents for this session not found. Please upload them first.")
        raise HTTPException(status_code=500, detail=f"S3 Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
