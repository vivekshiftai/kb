#!/bin/bash

# Quick Installation Script for KB API
# This script installs all required dependencies using the exact pip command

set -e  # Exit on any error

echo "🚀 Starting quick installation..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libmagic1 \
    build-essential \
    git \
    curl \
    wget

# Install all Python dependencies with exact versions
echo "📚 Installing Python dependencies..."
pip3 install "mineru[core]" huggingface_hub sentence-transformers chromadb pdf2image PyMuPDF sglang

# Install additional FastAPI dependencies
echo "🌐 Installing FastAPI dependencies..."
pip3 install fastapi uvicorn[standard] python-multipart

# Install other required dependencies
echo "🔧 Installing additional dependencies..."
pip3 install pydantic pydantic-settings structlog python-dotenv aiofiles python-magic starlette

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements.txt in virtual environment
echo "📋 Installing requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p uploads minieu_output chroma_db output logs

# Set up environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment file..."
    cp env.example .env
    echo "⚠️ Please edit .env file and set your OpenAI API key"
else
    echo "✅ Environment file already exists"
fi

# Test installation
echo "🧪 Testing installation..."
if mineru --version &> /dev/null; then
    echo "✅ MinerU installed successfully"
else
    echo "❌ MinerU installation failed"
    exit 1
fi

echo "🎉 Quick installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and set your OpenAI API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the application: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🔧 Test commands:"
echo "- MinerU: mineru --help"
echo "- API: curl http://localhost:8000/health/"
echo "- Docs: http://localhost:8000/docs"
