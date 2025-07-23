# DocuMentor AI

**A Specialized Q&A Chatbot with Fine-Tuning and RAG**

## Project Overview

DocuMentor AI is a cost-free generative AI chatbot designed to help student researchers quickly synthesize information from research papers. The system combines fine-tuning and Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers about research methodologies and conclusions.

### User Story
*"As a student researcher, I want to upload several new research papers and ask specific, technical questions about their methodologies and conclusions to quickly synthesize information for my literature review."*

## Architecture

```
User Upload → Document Processing → Local Vector Store → Fine-Tuned LLM → User Interface
```

### High-Level System Components

1. **Data Source**: arXiv API for Machine Learning/Computer Vision research papers
2. **Document Processing**: PyMuPDF for PDF parsing and text extraction
3. **Vector Database**: ChromaDB/FAISS for local storage
4. **Language Model**: Fine-tuned Llama 3 8B or Mistral 7B Instruct
5. **Backend API**: FastAPI for document upload and question handling
6. **Frontend**: Streamlit for user interface
7. **Deployment**: Hugging Face Spaces or Vercel (free tier)

## Project Phases

- **Phase 0**: Foundation & Scoping ✅ (Current)
- **Phase 1**: Data Acquisition & Preparation (Weeks 2-3)
- **Phase 2**: Model Fine-Tuning & Evaluation (Weeks 4-5)
- **Phase 3**: RAG Pipeline Development (Weeks 6-7)
- **Phase 4**: API & UI Development (Weeks 8-9)
- **Phase 5**: Deployment & Sharing (Weeks 10-12)
- **Phase 6**: Documentation & Portfolio Presentation (Week 13)

## Technology Stack

### Core Libraries
- **Data Processing**: `arxiv`, `PyMuPDF`, `pandas`
- **ML/AI**: `transformers`, `torch`, `langchain`, `sentence-transformers`
- **Vector Storage**: `chromadb`, `faiss-cpu`
- **API**: `fastapi`, `uvicorn`
- **UI**: `streamlit`
- **Development**: `jupyter`, `pytest`

### Cloud Services (Free Tier)
- **Model Hosting**: Hugging Face Hub
- **Deployment**: Hugging Face Spaces / Vercel
- **Training**: Google Colab / Kaggle Kernels
- **Version Control**: GitHub with Git LFS

## Development Setup

### Prerequisites
- Python 3.8+
- Git with Git LFS
- NVIDIA GPU (optional, for local training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/DocuMentor-AI.git
cd DocuMentor-AI
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Jupyter for development:
```bash
jupyter lab
```

## Domain Focus

**Research Domain**: Machine Learning and Computer Vision papers from arXiv
- **Rationale**: Dense, technical content with consistent structure
- **Data Source**: arXiv API (free, high-quality, extensive)
- **Target Papers**: 500-1000 most relevant papers in ML/CV

## Project Goals

1. **Cost-Free Implementation**: Utilize only free-tier services and open-source tools
2. **Production Quality**: Build a deployable, shareable application
3. **Educational Value**: Document the entire process for portfolio presentation
4. **Performance**: Demonstrate measurable improvement over base models

## Getting Started

1. Follow the installation instructions above
2. Review the project plan in `Project Plan.md`
3. Start with Phase 1: Data Acquisition & Preparation

## Contributing

This is a learning project. Feel free to fork and adapt for your own research domain!

## License

MIT License - See LICENSE file for details 