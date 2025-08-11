# RAG PDF Processing API - Backend Only

A high-performance FastAPI backend application that processes PDF documents and provides intelligent querying capabilities using Pinecone vector database and OpenAI's GPT models.

## 🚀 Features

- **PDF Upload & Processing**: Upload PDF files and extract text, images, and metadata
- **Vector Database Storage**: Store processed content in Pinecone with PDF-specific namespaces
- **Intelligent Querying**: Query specific PDFs using natural language with OpenAI's LLM
- **Image Extraction & Serving**: Extract and serve images from PDF documents via static URLs
- **RESTful API**: Clean, well-documented API endpoints with OpenAPI/Swagger docs
- **Performance Optimized**: Target response times < 1s with async processing
- **Comprehensive Logging**: Structured JSON logging for monitoring and debugging
- **Background Processing**: Non-blocking PDF processing with status tracking

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- 4GB+ RAM recommended for PDF processing

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1500
OPENAI_TEMPERATURE=0.7

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=pdf-rag-index

# File Processing
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
MAX_FILE_SIZE=52428800

# Processing Configuration
CHUNK_MAX_LENGTH=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=10

# Logging
LOG_LEVEL=INFO
```

### 3. Create Required Directories

```bash
mkdir -p uploads outputs
```

## 🚀 Running the Application

### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📚 API Endpoints

### 🏠 Root & Health
- `GET /` - API information and available endpoints
- `GET /health/` - Comprehensive health check

### 📄 PDF Management
- `POST /upload-pdf/` - Upload and process PDF files
- `GET /pdfs/` - List all processed PDFs with metadata
- `DELETE /pdfs/{filename}` - Delete PDF and all associated data

### 🔍 Querying
- `POST /query/` - Query specific PDFs with natural language

### 🖼️ Static Files
- `GET /images/{path}` - Serve extracted PDF images

## 💡 Usage Examples

### Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "PDF uploaded successfully. Processing started in background.",
  "pdf_filename": "your_document.pdf",
  "processing_status": "processing"
}
```

### List Processed PDFs
```bash
curl -X GET "http://localhost:8000/pdfs/"
```

**Response:**
```json
{
  "pdfs": [
    {
      "filename": "manual.pdf",
      "chunk_count": 25,
      "file_size": 2048576,
      "upload_date": "2024-01-15T10:30:00"
    }
  ],
  "total_count": 1
}
```

### Query a PDF
```bash
curl -X POST "http://localhost:8000/query/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "pdf_filename": "manual.pdf",
       "query": "How do I install the conveyor belt?",
       "max_results": 5
     }'
```

**Response:**
```json
{
  "pdf_filename": "manual.pdf",
  "query": "How do I install the conveyor belt?",
  "answer": "To install the conveyor belt, follow these steps: 1. Remove the old belt...",
  "results": [
    {
      "heading": "Installation Instructions",
      "text": "Step-by-step installation guide...",
      "score": 0.95,
      "page_number": 1,
      "images": [
        {
          "filename": "installation_diagram.png",
          "url": "/images/manual/page_1_img_1.png",
          "page_number": 1
        }
      ]
    }
  ],
  "total_matches": 3,
  "processing_time": 0.85
}
```

## 🏗️ Architecture

### Core Components

- **FastAPI Application** (`main.py`): Main API server with all endpoints
- **PDF Processor** (`services/pdf_processor.py`): Extracts text, images, and metadata
- **Vector Store** (`services/vector_store.py`): Manages Pinecone operations
- **Embedding Service** (`services/embeddings.py`): Generates text embeddings
- **OpenAI Client** (`services/openai_client.py`): Handles LLM interactions

### Data Flow

1. **PDF Upload** → File validation → Background processing
2. **Processing** → Text extraction → Chunking → Embedding generation → Pinecone storage
3. **Querying** → Query embedding → Similarity search → Context retrieval → LLM response

### Vector Database Design

- **Pinecone Index**: Single index with PDF-specific namespaces
- **Namespace Format**: `pdf_{filename_without_extension}`
- **Metadata**: Includes heading, text, images, page numbers, chunk indices
- **Embeddings**: 384-dimensional vectors from sentence-transformers

## 🧪 Testing

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py -v
```

## 📊 Performance Features

- **Async Processing**: Background PDF processing for non-blocking uploads
- **Efficient Chunking**: Smart text chunking with sentence boundary detection
- **Batch Operations**: Batch vector upserts to Pinecone
- **Response Optimization**: Target < 1s query response times
- **Memory Management**: Efficient handling of large PDFs

## 🔧 Configuration Options

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PINECONE_API_KEY` | Pinecone API key | Required |
| `PINECONE_ENVIRONMENT` | Pinecone environment | `us-east-1-aws` |
| `CHUNK_MAX_LENGTH` | Maximum chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `MAX_FILE_SIZE` | Maximum PDF file size | `50MB` |
| `LOG_LEVEL` | Logging level | `INFO` |

## 📝 Logging

The application uses structured JSON logging with:
- Request/response tracking
- Processing pipeline monitoring
- Error tracking with context
- Performance metrics

Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`

## 🔒 Security Features

- **File Validation**: Strict PDF file validation
- **Size Limits**: Configurable file size limits
- **Input Sanitization**: Clean filenames and text inputs
- **CORS Configuration**: Configurable CORS settings
- **Error Handling**: Secure error messages without sensitive data

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```env
LOG_LEVEL=WARNING
MAX_CONCURRENT_PROCESSING=2
CLEANUP_INTERVAL_DAYS=30
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Pinecone Connection Errors**
   - Verify API key and environment settings
   - Check network connectivity
   - Ensure index exists or can be created

2. **PDF Processing Failures**
   - Check PDF file integrity
   - Verify file size limits
   - Ensure sufficient disk space

3. **OpenAI API Errors**
   - Verify API key validity
   - Check rate limits and quotas
   - Monitor token usage

### Debug Mode
```bash
LOG_LEVEL=DEBUG uvicorn main:app --reload
```

For additional support, please check the logs and API documentation at `/docs`.