# RAG PDF Processing API

A high-performance FastAPI application that processes PDF documents and provides intelligent querying capabilities using Pinecone vector database and OpenAI's GPT models.

## Features

- **PDF Upload & Processing**: Upload PDF files and extract text, images, and metadata
- **Vector Database Storage**: Store processed content in Pinecone for semantic search with PDF-specific collections
- **Intelligent Querying**: Query specific PDFs using natural language with OpenAI's LLM
- **Image Extraction**: Extract and serve images from PDF documents
- **RESTful API**: Clean, well-documented API endpoints
- **Performance Optimized**: Target response times < 1s
- **Comprehensive Logging**: Structured logging for monitoring and debugging
- **Test Coverage**: Comprehensive test suite with pytest

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository and navigate to the project directory**

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=pdf-rag-index
   
   UPLOAD_DIR=./uploads
   OUTPUT_DIR=./outputs
   LOG_LEVEL=INFO
   ```

5. **Create necessary directories**:
   ```bash
   mkdir -p uploads outputs
   ```

## Usage

### Starting the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### API Endpoints

#### 1. Upload PDF
```bash
POST /upload-pdf/
```
Upload a PDF file for processing and storage in vector database.

**Example**:
```bash
curl -X POST "http://localhost:8000/upload-pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

**Response**:
```json
{
  "success": true,
  "message": "PDF uploaded successfully, processing started",
  "pdf_filename": "your_document.pdf",
  "processing_status": "processing"
}
```

#### 2. List Processed PDFs
```bash
GET /pdfs/
```
Get a list of all processed PDF filenames.

**Example**:
```bash
curl -X GET "http://localhost:8000/pdfs/"
```

**Response**:
```json
{
  "pdfs": [
    {
      "filename": "document1.pdf",
      "chunk_count": 25
    },
    {
      "filename": "document2.pdf", 
      "chunk_count": 18
    }
  ],
  "total_count": 2
}
```

#### 3. Query PDF
```bash
POST /query/
```
Ask questions about a specific uploaded PDF.

**Example**:
```bash
curl -X POST "http://localhost:8000/query/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "pdf_filename": "your_document.pdf",
       "query": "How do I install the conveyor belt?",
       "max_results": 5
     }'
```

**Response**:
```json
{
  "pdf_filename": "your_document.pdf",
  "query": "How do I install the conveyor belt?",
  "answer": "To install the conveyor belt, follow these steps: 1. Remove the old belt...",
  "results": [
    {
      "heading": "Installation Instructions",
      "text": "Step-by-step installation guide...",
      "score": 0.95,
      "images": [
        {
          "filename": "installation_diagram.png",
          "url": "/images/your_document/page_1_img_1.png",
          "page_number": 1
        }
      ]
    }
  ],
  "total_matches": 3,
  "processing_time": 0.85
}
```

#### 4. Delete PDF
```bash
DELETE /pdfs/{pdf_filename}
```
Delete a PDF and all its associated data.

#### 5. Health Check
```bash
GET /health/
```
Check if the service is running properly.

## Architecture

### Key Components

- **FastAPI Application** (`main.py`): Main API server with endpoints
- **PDF Processor** (`services/pdf_processor.py`): Extracts text, images, and metadata from PDFs
- **Vector Store** (`services/vector_store.py`): Manages Pinecone operations with PDF-specific namespaces
- **Embedding Service** (`services/embeddings.py`): Generates text embeddings using sentence-transformers
- **OpenAI Client** (`services/openai_client.py`): Handles LLM interactions for answer generation

### Data Flow

1. **PDF Upload**: User uploads PDF → File validation → Background processing
2. **Processing**: Text extraction → Chunking → Embedding generation → Pinecone storage
3. **Querying**: Query embedding → Similarity search in PDF namespace → Context retrieval → LLM response generation

### Vector Database Design

- **Pinecone Index**: Single index with PDF-specific namespaces
- **Namespace Format**: `pdf_{filename_without_extension}`
- **Metadata**: Includes heading, text snippet, images, page numbers, and chunk indices
- **Embeddings**: 384-dimensional vectors from sentence-transformers

## Configuration

Key environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key  
- `PINECONE_ENVIRONMENT`: Pinecone environment (e.g., "us-west1-gcp")
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `CHUNK_MAX_LENGTH`: Maximum chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_SEARCH_RESULTS`: Maximum search results (default: 10)

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py -v
```

## Performance Optimization

- **Async Processing**: Background PDF processing for non-blocking uploads
- **Efficient Chunking**: Smart text chunking with sentence boundary detection
- **Batch Operations**: Batch vector upserts to Pinecone
- **Caching**: Singleton pattern for services and settings
- **Response Time**: Optimized for < 1s query response times

## Logging

Structured JSON logging with:
- Request/response tracking
- Processing pipeline monitoring
- Error tracking with context
- Performance metrics

Log levels: DEBUG, INFO, WARNING, ERROR

## Error Handling

- **Validation Errors**: Clear error messages for invalid inputs
- **Rate Limiting**: Graceful handling of API rate limits
- **File Errors**: Proper handling of corrupted or invalid PDFs
- **Service Errors**: Fallback responses for service failures

## Security

- **File Validation**: Strict PDF file validation
- **Size Limits**: Configurable file size limits
- **Input Sanitization**: Clean filenames and text inputs
- **API Key Management**: Secure environment variable handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License.