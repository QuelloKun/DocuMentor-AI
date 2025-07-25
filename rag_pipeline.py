# rag_pipeline.py

import os
import tempfile
import warnings
from typing import List, Optional

import torch

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress HTTP retry logs from model downloading
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FINETUNED_MODEL = "models/documentor-qwen-3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_llm() -> HuggingFacePipeline:
    """Load the fine-tuned model for inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this model. Please ensure you have a GPU with CUDA support.")
    
    print(f"Using device: cuda (GPUs available: {torch.cuda.device_count()})")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        do_sample=True
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def create_vector_store(pdf_files: List) -> Optional[FAISS]:
    """Create vector store from PDF files."""
    if not pdf_files:
        return None
    
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in pdf_files:
            temp_filepath = os.path.join(temp_dir, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(documents=splits, embedding=embeddings)

def create_rag_chain(llm: HuggingFacePipeline, vector_store: FAISS) -> Optional[RetrievalQA]:
    """Create the RetrievalQA chain."""
    if not vector_store:
        return None
    
    prompt_template = """You are a helpful AI assistant that answers questions based on the provided documents. Use the following context to provide a clear, comprehensive answer.

INSTRUCTIONS:
- Give a direct, well-structured answer based on the provided context
- If the information is incomplete, say so and provide what you can find
- If the context doesn't contain relevant information, clearly state that you cannot find the answer in the provided documents
- Format your response clearly with proper structure (use bullet points, numbered lists, or paragraphs as appropriate)
- Do not repeat the question in your answer
- Be concise but thorough

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )