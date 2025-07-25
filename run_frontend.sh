#!/bin/bash

# DocuMentor AI Frontend Runner
# Usage: ./run_frontend.sh

echo "🖥️  DocuMentor AI Frontend Runner"
echo "=================================="
echo ""

check_docker() {
    if ! docker info &> /dev/null; then
        echo "❌ Error: Docker is not running or not accessible"
        echo "   Please start Docker and try again"
        exit 1
    fi
}

check_docker

echo "🏗️  Building frontend container..."
if docker build -f Dockerfile.frontend -t documentor-frontend .; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting frontend container..."
    echo "📊 Make sure your backend is running at: http://localhost:8000"
    echo ""
    
    mkdir -p data/sessions
    
    echo "🔗 Connecting to backend at localhost:8000"
    docker run --rm --name documentor-frontend \
        --network="host" \
        -v "$(pwd)/data/sessions":/app/data/sessions \
        -e BACKEND_URL="http://localhost:8000" \
        documentor-frontend
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🌐 Frontend was running at: http://localhost:8501"
echo "To stop the container, press Ctrl+C" 