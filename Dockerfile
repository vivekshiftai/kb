# Use Ubuntu as base image for Minieu compatibility
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Install MinerU with core dependencies (updated to v2.1.0)
RUN pip3 install "mineru[core]==2.1.0"

# Install additional required dependencies
RUN pip3 install huggingface_hub==0.20.3 sentence-transformers==2.2.2 chromadb==1.0.16 pdf2image==1.17.0 PyMuPDF==1.26.3 sglang==0.1.0

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/minieu_output /app/chroma_db /app/output

# Set permissions
RUN chmod +x /app/install_minieu.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
