# RAG PDF Processing API with Minieu

A high-performance FastAPI application for PDF processing and intelligent querying using Minieu for document extraction and ChromaDB for vector storage.

## 🚀 Features

- **PDF Processing**: Uses Minieu for advanced PDF text and image extraction
- **Vector Storage**: ChromaDB for efficient semantic search
- **AI-Powered Queries**: OpenAI integration for intelligent responses
- **Image Handling**: Automatic image extraction and serving
- **Rules Generation**: IoT device rules, maintenance data, and safety precautions
- **RESTful API**: Complete FastAPI documentation and testing interface
- **Production Ready**: Comprehensive logging and error handling

## 📋 Prerequisites

- Python 3.8+
- Ubuntu 22.04+ (recommended) or macOS
- OpenAI API key
- 4GB+ RAM (for Minieu processing)
- 10GB+ disk space

## 🛠️ Quick Setup

### Option 1: Simple Setup Script

```bash
# Clone the repository
git clone <your-repo-url>
cd kb

# Run the setup script
python setup.py
```

### Option 2: Manual Setup

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
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
```

**macOS:**
```bash
brew install \
    python3 \
    poppler \
    tesseract \
    tesseract-lang \
    libmagic \
    git \
    curl \
    wget
```

#### 2. Install Minieu and Dependencies

```bash
# Install Minieu with core dependencies
pip3 install "mineru[core]==2.1.0"

# Install additional required dependencies
pip3 install huggingface_hub==0.20.3 sentence-transformers==2.2.2 chromadb==1.0.16 pdf2image==1.17.0 PyMuPDF==1.26.3
```

#### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your settings
nano .env
```

**Required environment variables:**
```env
OPENAI_API_KEY=your_openai_api_key_here
MINIEU_OUTPUT_DIR=./minieu_output
UPLOAD_DIR=./uploads
```

#### 5. Create Directories

```bash
mkdir -p uploads minieu_output chroma_db output logs
```

## 🚀 Running the Application

### Development Mode

```bash
# Option 1: Using the run script (recommended)
python run.py

# Option 2: Using uvicorn directly
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Using gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Usage

### 1. Upload and Process PDF

```bash
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### 2. Query PDF Content

```bash
curl -X POST "http://localhost:8000/query/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "pdf_filename": "your_document.pdf",
       "query": "How to maintain the equipment?",
       "max_results": 5
     }'
```

### 3. Generate Rules

```bash
curl -X POST "http://localhost:8000/rules/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf" \
     -F "chunk_size=30" \
     -F "rule_types=monitoring,maintenance,alert"
```

### 4. List Processed PDFs

```bash
curl -X GET "http://localhost:8000/pdfs/" \
     -H "accept: application/json"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `MINIEU_OUTPUT_DIR` | Minieu output directory | `./minieu_output` |
| `UPLOAD_DIR` | PDF upload directory | `./uploads` |
| `MAX_FILE_SIZE` | Maximum file size (bytes) | `52428800` (50MB) |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CHUNK_MAX_LENGTH` | Maximum chunk length | `1000` |
| `MAX_SEARCH_RESULTS` | Maximum search results | `10` |

### Minieu Configuration

The application automatically calls Minieu to process PDFs. Minieu settings can be configured in the `MinieuProcessor` class:

- **Timeout**: 300 seconds (5 minutes)
- **Retries**: 3 attempts
- **Output Format**: Markdown + images

## 📁 Project Structure

```
kb/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── setup.py               # Simple setup script
├── run.py                 # Simple run script
├── config/
│   └── settings.py       # Application settings
├── models/
│   └── schemas.py        # Pydantic models
├── services/
│   ├── minieu_processor.py  # Minieu integration
│   ├── pdf_processor.py     # PDF processing
│   ├── openai_client.py     # OpenAI integration
│   └── rules_generator.py   # Rules generation
├── utils/
│   ├── file_utils.py     # File utilities
│   └── helpers.py        # Helper functions
├── uploads/               # Sample PDF files
├── outputs/               # Processed outputs and images
├── env.example            # Environment variables template
└── README.md             # This file
```

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health/
```

### Debug Endpoints

```bash
# Check Minieu status
curl http://localhost:8000/debug/minieu-status/

# Test Minieu processing
curl -X POST "http://localhost:8000/debug/process-with-minieu/" \
     -H "Content-Type: application/json" \
     -d '{"pdf_filename": "test.pdf"}'

# Check images
curl http://localhost:8000/debug/images/
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔍 Troubleshooting

### Common Issues

1. **Minieu not found**
   ```bash
   pip3 install "mineru[core]==2.1.0"
   mineru --version
   ```

2. **Permission denied**
   ```bash
   # Make sure you have write permissions to the directory
   ```

3. **OpenAI API errors**
   - Check your API key in `.env`
   - Verify API key has sufficient credits

4. **ChromaDB connection issues**
   ```bash
   # Check ChromaDB installation
   python -c "import chromadb; print(chromadb.__version__)"
   ```

### Logs

```bash
# Application logs
tail -f app.log

# Minieu processing logs
tail -f logs/minieu.log
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Systemd Service

Create `/etc/systemd/system/kb-api.service`:

```ini
[Unit]
Description=KB PDF Processing API
After=network.target

[Service]
Type=exec
User=your_user
WorkingDirectory=/path/to/kb
Environment=PATH=/path/to/kb/venv/bin
ExecStart=/path/to/kb/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable kb-api
sudo systemctl start kb-api
sudo systemctl status kb-api
```

### Environment Variables for Production

```env
DEBUG=false
LOG_LEVEL=WARNING
OPENAI_API_KEY=your_production_key
SECRET_KEY=your_secure_secret_key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the `/docs` endpoint
- **Debug**: Use the debug endpoints for troubleshooting

## 🔄 Changelog

### v2.0.0
- Added Minieu integration for PDF processing
- Improved image handling and serving
- Enhanced rules generation with safety precautions
- Improved error handling and logging
- Updated to MinerU v2.1.0 with enhanced features