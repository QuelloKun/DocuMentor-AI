# DocuMentor AI

An intelligent document analysis system that combines fine-tuned language models with RAG (Retrieval-Augmented Generation) to answer questions about research papers.

## Features

- **Fine-tuned LLM**: Qwen2.5-3B-Instruct optimized for document Q&A
- **RAG Pipeline**: Efficient document retrieval and context-aware responses
- **Web Interface**: Clean Streamlit frontend with session management
- **Dockerized**: Production-ready containerized deployment
- **GPU Required**: Optimized for NVIDIA GPU inference with CUDA support
- **Persistent Sessions**: Chat history and document processing saved between sessions

## Interface

![DocuMentor AI Interface](assets/app-interface.png)

*Clean, intuitive interface for document upload and intelligent Q&A*

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │────│    Backend       │────│   ML Pipeline   │
│   (Streamlit)   │    │   (FastAPI)      │    │   (Qwen2.5-3B)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────────┐
                       │  Vector Store    │
                       │    (FAISS)       │
                       └──────────────────┘
```

## Prerequisites

- **Linux** system (Ubuntu/Debian recommended)
- **NVIDIA GPU** with 8GB+ VRAM and CUDA support
- **Docker** with **NVIDIA Container Toolkit**
- **Git** with **Git LFS** (for model files)

### Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Install Git LFS

```bash
sudo apt install git-lfs
git lfs install
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/QuelloKun/DocuMentor-AI.git
cd DocuMentor-AI

# Pull model files
git lfs pull

# Start backend (GPU required)
./run_backend.sh

# In another terminal, start frontend
./run_frontend.sh

# Access the application at http://localhost:8501
```

## Usage

### Running the Application

1. **Start Backend**: Run `./run_backend.sh` to start the GPU-accelerated backend
2. **Start Frontend**: Run `./run_frontend.sh` to start the web interface
3. **Access**: Open http://localhost:8501 in your browser

### Using the Interface

1. **Session Management**: 
   - Create new sessions or load previous ones from the sidebar
   - Rename or delete sessions as needed
   - All chat history and document processing is automatically saved

2. **Upload Documents**: 
   - Use the sidebar to upload PDF files
   - Click "Process Documents" to analyze your files
   - Document processing state is remembered per session

3. **Ask Questions**: 
   - Use the chat interface to query your documents
   - Receive contextual responses based on document content
   - All conversations are automatically saved

## Technical Details

- **Base Model**: Qwen2.5-3B-Instruct with LoRA fine-tuning
- **Embeddings**: all-MiniLM-L6-v2 for document vectorization
- **Vector Store**: FAISS for efficient similarity search
- **Text Processing**: LangChain for document splitting and RAG
- **Deployment**: Separate Docker containers for frontend and backend
- **Persistence**: JSON-based session storage with automatic saving

## API Documentation

With the backend running, visit `http://localhost:8000/docs` for interactive API documentation.

## Troubleshooting

### GPU Issues
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Verify Docker GPU support: `docker run --gpus all nvidia/cuda:11.8-devel-ubuntu22.04 nvidia-smi`
- Check CUDA compatibility with your GPU

### Memory Issues
- Minimum 8GB VRAM required
- Close other GPU applications before running
- Monitor GPU memory: `nvidia-smi -l 1`

### Connection Issues
- Ensure backend is running before starting frontend
- Check if ports 8000 and 8501 are available
- Verify Docker containers are running: `docker ps`

## Future Enhancements

### Planned Features
- [ ] **Advanced Search**: Semantic search across multiple document sessions
- [ ] **AMD GPU Support**: Add ROCm support for AMD graphics cards
- [ ] **Export Functionality**: Export chat history and processed documents

## License

This project uses open-source components under permissive licenses. See individual package licenses for details. 
