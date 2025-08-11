# RAG PDF Processing API - Backend Only

A high-performance FastAPI backend application that processes PDF documents and provides intelligent querying capabilities using Pinecone vector database and OpenAI's GPT models.

## 🚀 Features

- **PDF Upload & Processing**: Upload PDF files and extract text, images, and metadata
- **Vector Database Storage**: Store processed content in Pinecone with graceful fallback to ChromaDB
- **Intelligent Querying**: Query specific PDFs using natural language with OpenAI's LLM
- **IoT Rules Generation**: Generate IoT device rules and maintenance data from PDF content
- **Advanced PDF Processing**: ToC-based semantic chunking with separate text and image collections
- **Image Extraction & Serving**: Extract and serve images from PDF documents via static URLs
- **RESTful API**: Clean, well-documented API endpoints with OpenAPI/Swagger docs
- **Performance Optimized**: Target response times < 1s with async processing
- **Comprehensive Logging**: Structured JSON logging for monitoring and debugging
- **Background Processing**: Non-blocking PDF processing with status tracking
- **Graceful Fallback**: Automatic fallback from Pinecone to ChromaDB on connection issues

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment (optional - falls back to ChromaDB)
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

# Pinecone Configuration (Optional - falls back to ChromaDB if not available)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=pdf-rag-index

# ChromaDB Configuration (Fallback vector database)
CHROMA_PERSIST_DIRECTORY=./chroma_db

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
mkdir -p uploads outputs chroma_db
```

Or run the helper script:
```bash
python create_dirs.py
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

## 🧪 Testing

### Test Logging Configuration
```bash
python test_logging.py
```

### Test PDF Processing Logging
```bash
python test_pdf_logging.py
```

### Test Rules API
```bash
python test_rules_api.py
```

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

### ⚙️ Rules Generation
- `POST /rules/` - Generate IoT device rules and maintenance data from PDF content

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

### Generate IoT Rules and Maintenance Data

**Option 1: Using an already uploaded PDF**
```bash
curl -X POST "http://localhost:8000/rules/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "pdf_filename=equipment_manual.pdf" \
     -F "chunk_size=10" \
     -F "rule_types=monitoring" \
     -F "rule_types=maintenance" \
     -F "rule_types=alert"
```

**Option 2: Direct PDF upload for rules generation**
```bash
curl -X POST "http://localhost:8000/rules/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@equipment_manual.pdf" \
     -F "chunk_size=10" \
     -F "rule_types=monitoring" \
     -F "rule_types=maintenance" \
     -F "rule_types=alert"
```

**Response:**
```json
{
  "pdf_filename": "equipment_manual.pdf",
  "total_pages": 45,
  "processed_chunks": 5,
  "iot_rules": [
    {
      "device_name": "Temperature Sensor T1",
      "rule_type": "monitoring",
      "condition": "Temperature exceeds 85°C",
      "action": "Send alert to maintenance team",
      "priority": "high",
      "frequency": "hourly",
      "description": "Monitor equipment temperature to prevent overheating"
    },
    {
      "device_name": "Conveyor Belt Motor",
      "rule_type": "maintenance",
      "condition": "Operating hours reach 1000",
      "action": "Schedule preventive maintenance",
      "priority": "medium",
      "frequency": "weekly",
      "description": "Regular maintenance schedule for motor components"
    }
  ],
  "maintenance_data": [
    {
      "component_name": "Filter Assembly",
      "maintenance_type": "preventive",
      "frequency": "Every 3 months",
      "last_maintenance": "2024-01-15",
      "next_maintenance": "2024-04-15",
      "description": "Replace air filters to maintain optimal performance"
    }
  ],
  "processing_time": 15.67,
  "summary": "Generated 8 IoT monitoring rules and 5 maintenance schedules from 45-page equipment manual."
}
```

## 🏗️ Architecture

### Core Components

- **FastAPI Application** (`main.py`): Main API server with all endpoints
- **PDF Processor** (`services/pdf_processor.py`): Extracts text, images, and metadata with ToC-based chunking
- **Vector Store** (`services/vector_store.py`): Manages Pinecone operations with ChromaDB fallback
- **Rules Generator** (`services/rules_generator.py`): Generates IoT rules and maintenance data
- **Embedding Service** (`services/embeddings.py`): Generates text embeddings
- **OpenAI Client** (`services/openai_client.py`): Handles LLM interactions

### Data Flow

1. **PDF Upload** → File validation → Background processing
2. **Processing** → Text extraction → Chunking → Embedding generation → Pinecone storage
3. **Querying** → Query embedding → Similarity search → Context retrieval → LLM response

### Vector Database Design

- **Primary**: Pinecone Index with PDF-specific namespaces
- **Fallback**: ChromaDB with separate collections for text and images
- **Namespace Format**: `pdf_{filename_without_extension}`
- **Metadata**: Includes heading, text, images, page numbers, chunk indices, tables count
- **Text Embeddings**: 384-dimensional vectors from sentence-transformers
- **Image Embeddings**: CLIP embeddings for visual content (ChromaDB only)
- **Chunking Strategy**: ToC-based semantic chunking for better context preservation

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

# Test API endpoints using curl or Postman
curl -X GET "http://localhost:8000/health/"

# Test PDF upload
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"

# Test querying
curl -X POST "http://localhost:8000/query/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"pdf_filename": "your_document.pdf", "query": "What is the maintenance schedule?"}'

# Test rules generation
curl -X POST "http://localhost:8000/rules/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

## 📊 Performance Features

- **Async Processing**: Background PDF processing for non-blocking uploads
- **Advanced Chunking**: ToC-based semantic chunking for better context preservation
- **Dual Vector Storage**: Pinecone with ChromaDB fallback for reliability
- **Separate Collections**: Text and image embeddings stored separately in ChromaDB
- **Batch Operations**: Batch vector upserts to vector databases
- **Response Optimization**: Target < 1s query response times
- **Memory Management**: Efficient handling of large PDFs
- **Image Processing**: CLIP embeddings for visual content analysis

## 🔧 Configuration Options

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PINECONE_API_KEY` | Pinecone API key | Optional (falls back to ChromaDB) |
| `PINECONE_ENVIRONMENT` | Pinecone environment | `us-east-1-aws` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage directory | `./chroma_db` |
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

## 🤖 Rules Generation Features

The application includes advanced IoT rules generation capabilities:

### **IoT Device Rules**
- **Monitoring Rules**: Device state monitoring and threshold detection
- **Maintenance Rules**: Preventive and predictive maintenance schedules
- **Alert Rules**: Condition-based alerting and notifications
- **Control Rules**: Automated device control and response actions

### **Maintenance Data Extraction**
- **Component Information**: Device and component identification
- **Maintenance Schedules**: Frequency and timing requirements
- **Service History**: Last and next maintenance dates
- **Procedures**: Detailed maintenance procedures and requirements

### **Processing Capabilities**
- **Chunk-based Processing**: Processes PDFs in configurable page chunks (default: 10 pages)
- **Dual Input Methods**: Direct file upload or use existing uploaded PDFs
- **Intelligent Deduplication**: Removes duplicate rules and maintenance entries
- **Comprehensive Summaries**: Provides overview of generated content
- **Multiple Rule Types**: Configurable rule type generation

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
   - Application will automatically fall back to ChromaDB

2. **PDF Processing Failures**
   - Check PDF file integrity
   - Verify file size limits
   - Ensure sufficient disk space
   - Check for image extraction errors (CMYK color space issues)

3. **OpenAI API Errors**
   - Verify API key validity
   - Check rate limits and quotas
   - Monitor token usage

4. **Rules Generation Issues**
   - Ensure PDF contains relevant technical content
   - Check OpenAI API quota for large documents
   - Verify chunk size settings for optimal processing

### Debug Mode
```bash
LOG_LEVEL=DEBUG uvicorn main:app --reload
```

### PDF Processing Errors

If you're experiencing errors when adding PDFs:

1. **Check the application logs:**
   ```bash
   tail -f app.log
   ```

2. **Verify directory structure:**
   ```bash
   ls -la uploads/ outputs/ chroma_db/
   ```

3. **Common PDF processing issues:**
   - **Missing directories**: Ensure `uploads/`, `outputs/`, and `chroma_db/` directories exist
   - **Import errors**: Check if all dependencies are installed with `pip install -r requirements.txt`
   - **Vector store issues**: Verify ChromaDB/Pinecone configuration in `.env`
   - **Logging errors**: Ensure structlog is properly configured

4. **If errors persist:**
   - Check the `app.log` file for detailed error messages
   - Verify your `.env` file configuration
   - Ensure sufficient disk space for uploads and vector storage

For additional support, please check the logs and API documentation at `/docs`.