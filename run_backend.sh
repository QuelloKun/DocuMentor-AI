#!/bin/bash

# DocuMentor AI Backend Runner (GPU Required)
# Usage: ./run_backend.sh

echo "🤖 DocuMentor AI Backend Runner"
echo "================================"
echo "⚠️  GPU with CUDA support required"
echo ""

check_docker() {
    if ! docker info &> /dev/null; then
        echo "❌ Error: Docker is not running or not accessible"
        echo "   Please start Docker and try again"
        exit 1
    fi
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

check_docker

if ! check_gpu; then
    echo "❌ Error: No GPU detected or nvidia-smi not available"
    echo "   This model requires CUDA-compatible GPU"
    echo "   Please ensure:"
    echo "   - You have an NVIDIA GPU"
    echo "   - CUDA drivers are installed"
    echo "   - nvidia-docker2 is installed"
    exit 1
fi

echo "✅ GPU detected! Building backend..."
if docker build -f Dockerfile.backend -t documentor-backend .; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting backend container with GPU support..."
    echo "📊 GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    echo ""
    
    docker run --rm --name documentor-backend \
        --gpus all \
        -p 8000:8000 \
        documentor-backend
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🌐 Backend was running at: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📖 API docs: http://localhost:8000/docs" 