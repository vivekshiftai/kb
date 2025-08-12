#!/bin/bash

# Minieu Installation Script for Ubuntu
# This script installs Minieu and all required system dependencies

set -e  # Exit on any error

echo "🚀 Starting Minieu installation..."

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

# Install MinerU (updated to v2.1.0)
echo "🤖 Installing MinerU v2.1.0..."
pip3 install mineru==2.1.0

# Verify Minieu installation
echo "✅ Verifying Minieu installation..."
if command -v mineru &> /dev/null; then
    echo "🎉 Minieu installed successfully!"
    mineru --version
else
    echo "❌ Minieu installation failed!"
    exit 1
fi

# Create virtual environment for the application
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Set up environment variables: cp .env.example .env"
echo "3. Configure your OpenAI API key in .env"
echo "4. Run the application: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🔧 To test Minieu:"
echo "   mineru --help"
echo "   mineru process your_document.pdf --output ./output"
