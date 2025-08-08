# PDF Processing API with Vector Database

A FastAPI application that processes PDF documents using MinerU, stores content in ChromaDB vector database, and provides intelligent querying capabilities using OpenAI's API.

## Features

- **PDF Upload & Processing**: Upload PDF files and extract text, images, and tables
- **Vector Database Storage**: Store processed content in ChromaDB for semantic search
- **Intelligent Querying**: Use OpenAI's LLM to answer questions about uploaded documents
- **Image & Table Extraction**: Handle complex document layouts with images and tables
- **RESTful API**: Clean API endpoints for integration

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- OpenAI API key

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

4. **Download MinerU models** (first time only):
   ```bash
   python scripts/download_models.py
   ```

5. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DEVICE_MODE=cuda  # or 'cpu' if no GPU available
   UPLOAD_DIR=./uploads
   OUTPUT_DIR=./outputs
   CHROMADB_DIR=./chromadb_storage
   ```

6. **Create necessary directories**:
   ```bash
   mkdir -p uploads outputs chromadb_storage models
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

#### 1. Upload and Process PDF
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

#### 2. Query Documents
```bash
POST /query/
```
Ask questions about uploaded documents.

**Example**:
```bash
curl -X POST "http://localhost:8000/query/" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I install the conveyor belt?"}'
```

#### 3. List Processed Documents
```bash
GET /documents/
```
Get a list of all processed documents.

#### 4. Health Check
```bash
GET /health/
```
Check if the service is running properly.

## Project Structure

```
pdf-processor-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── config/
│   ├── __init__.py
│   ├── settings.py        # Application settings
│   └── mineru_config.py   # MinerU configuration
├── models/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic models
│   └── responses.py       # Response models
├── services/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF processing logic
│   ├── vector_store.py    # ChromaDB operations
│   ├── openai_client.py   # OpenAI API client
│   └── embeddings.py      # Embedding generation
├── utils/
│   ├── __init__.py
│   ├── file_utils.py      # File handling utilities
│   └── helpers.py         # Helper functions
├── scripts/
│   ├── __init__.py
│   └── download_models.py # Model download script
├── uploads/               # Uploaded PDF files
├── outputs/               # Processed outputs
├── chromadb_storage/      # ChromaDB storage
└── models/                # Downloaded AI models
```

## Configuration

The application uses environment variables for configuration. Key settings:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DEVICE_MODE`: Set to 'cuda' for GPU acceleration or 'cpu' for CPU-only
- `UPLOAD_DIR`: Directory for uploaded files
- `OUTPUT_DIR`: Directory for processed outputs
- `CHROMADB_DIR`: Directory for ChromaDB storage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or switch to CPU mode
2. **Model Download Fails**: Check internet connection and run download script again
3. **OpenAI API Errors**: Verify your API key and quota

### Performance Tips

- Use GPU for faster processing if available
- Increase `VIRTUAL_VRAM_SIZE` in config for larger documents
- Process smaller batches for limited memory systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.