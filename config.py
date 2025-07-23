"""
Configuration file for DocuMentor AI project.
Contains all configurable parameters and settings.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# arXiv API settings
ARXIV_CONFIG = {
    "max_papers": 1000,
    "subjects": ["cs.CV", "cs.LG", "cs.AI"],  # Computer Vision, Machine Learning, AI
    "sort_by": "relevance",
    "sort_order": "descending"
}

# Model settings
MODEL_CONFIG = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",  # Alternative: "meta-llama/Llama-2-7b-chat-hf"
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 4096,
    "temperature": 0.7,
    "top_p": 0.9
}

# Fine-tuning settings
TRAINING_CONFIG = {
    "output_dir": str(MODELS_DIR / "fine_tuned"),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "max_seq_length": 2048,
    "use_peft": True,  # Parameter Efficient Fine-Tuning
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}

# RAG settings
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "vector_db": "chromadb",  # Options: "chromadb", "faiss"
    "similarity_top_k": 5,
    "embedding_dimension": 384  # For all-MiniLM-L6-v2
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "allowed_extensions": [".pdf", ".txt", ".md"]
}

# UI settings
UI_CONFIG = {
    "app_title": "DocuMentor AI",
    "app_description": "AI-powered research paper Q&A assistant",
    "max_upload_size": 50,  # MB
    "default_model_name": "DocuMentor-7B"
}

# Environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For evaluation/comparison if needed

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
} 