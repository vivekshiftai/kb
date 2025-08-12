#!/bin/bash

# Comprehensive Setup Script for RAG PDF Processing API with Minieu
# This script sets up the complete environment

set -e  # Exit on any error

echo "🚀 Starting comprehensive setup for RAG PDF Processing API..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

print_status "Detected OS: $OS"

# Update package list (Linux only)
if [[ "$OS" == "linux" ]]; then
    print_status "Updating package list..."
    sudo apt-get update
fi

# Install system dependencies
print_status "Installing system dependencies..."
if [[ "$OS" == "linux" ]]; then
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
        wget \
        docker.io \
        docker-compose
elif [[ "$OS" == "macos" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is required for macOS. Please install it first."
        exit 1
    fi
    
    brew install \
        python3 \
        poppler \
        tesseract \
        tesseract-lang \
        libmagic \
        git \
        curl \
        wget \
        docker \
        docker-compose
fi

# Install MinerU with core dependencies (updated to v2.1.0)
print_status "Installing MinerU v2.1.0 with core dependencies..."
pip3 install "mineru[core]==2.1.0"

# Install additional required dependencies
print_status "Installing additional dependencies..."
pip3 install huggingface_hub==0.20.3 sentence-transformers==2.2.2 chromadb==1.0.16 pdf2image==1.17.0 PyMuPDF==1.26.3 sglang==0.1.0

# Verify Minieu installation
if command -v mineru &> /dev/null; then
    print_success "Minieu installed successfully!"
    mineru --version
else
    print_error "Minieu installation failed!"
    exit 1
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating application directories..."
mkdir -p uploads minieu_output chroma_db output logs

# Set up environment file
if [ ! -f .env ]; then
    print_status "Creating environment file..."
    cp env.example .env
    print_warning "Please edit .env file and set your OpenAI API key"
else
    print_status "Environment file already exists"
fi

# Set permissions
print_status "Setting file permissions..."
chmod +x install_minieu.sh
chmod +x scripts/setup.sh

# Test Minieu
print_status "Testing Minieu installation..."
if mineru --help &> /dev/null; then
    print_success "Minieu is working correctly"
else
    print_error "Minieu test failed"
    exit 1
fi

# Test Python environment
print_status "Testing Python environment..."
python -c "import fastapi, chromadb, openai, structlog; print('All Python dependencies installed successfully')"

print_success "Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and set your OpenAI API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the application: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🔧 Alternative deployment options:"
echo "- Docker: docker-compose up -d"
echo "- Production: Use the provided nginx.conf and SSL certificates"
echo ""
echo "🧪 Test the setup:"
echo "- Minieu: mineru --help"
echo "- API: curl http://localhost:8000/health/"
echo "- Docs: http://localhost:8000/docs"
