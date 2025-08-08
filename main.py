from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from typing import List, Dict, Any
import logging

from models.schemas import QueryRequest, QueryResponse, DocumentInfo, HealthResponse
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.openai_client import OpenAIClient
from config.settings import get_settings
from utils.file_utils import ensure_directories, get_file_hash, cleanup_old_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Processing API",
    description="API for processing PDFs and querying content using vector databases",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings and services
settings = get_settings()
pdf_processor = PDFProcessor()
vector_store = VectorStore()
openai_client = OpenAIClient()

# Ensure required directories exist
ensure_directories()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await vector_store.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await vector_store.close()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/health/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if vector store is accessible
        await vector_store.health_check()
        
        # Check if OpenAI client is configured
        openai_status = openai_client.check_connection()
        
        return HealthResponse(
            status="healthy",
            message="All services are running",
            vector_store_status="connected",
            openai_status="connected" if openai_status else "error"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/upload-pdf/")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PDF file"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file hash for deduplication
        file_hash = get_file_hash(file_path)
        
        # Check if file already processed
        existing_doc = await vector_store.get_document_by_hash(file_hash)
        if existing_doc:
            return JSONResponse(
                content={
                    "message": "File already processed",
                    "document_id": existing_doc["id"],
                    "status": "existing"
                }
            )
        
        # Process PDF in background
        background_tasks.add_task(
            process_pdf_background,
            file_path,
            file.filename,
            file_hash
        )
        
        return JSONResponse(
            content={
                "message": "PDF upload successful, processing started",
                "filename": file.filename,
                "status": "processing"
            }
        )
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

async def process_pdf_background(file_path: str, filename: str, file_hash: str):
    """Background task to process PDF"""
    try:
        logger.info(f"Starting background processing for {filename}")
        
        # Process PDF with MinerU
        output_dir = await pdf_processor.process_pdf(file_path, filename)
        
        # Extract and store content in vector database
        document_id = await vector_store.store_document(
            file_path=file_path,
            filename=filename,
            file_hash=file_hash,
            output_dir=output_dir
        )
        
        logger.info(f"Successfully processed {filename} with ID: {document_id}")
        
        # Cleanup old files if needed
        cleanup_old_files(settings.UPLOAD_DIR, days=7)
        
    except Exception as e:
        logger.error(f"Error in background processing for {filename}: {e}")

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query processed documents"""
    try:
        # Search for relevant chunks
        search_results = await vector_store.search(
            query=request.question,
            n_results=request.max_results or 5
        )
        
        if not search_results["documents"] or not search_results["documents"][0]:
            return QueryResponse(
                question=request.question,
                answer="I couldn't find any relevant information in the uploaded documents.",
                sources=[],
                confidence_score=0.0
            )
        
        # Generate response using OpenAI
        response = await openai_client.generate_response(
            question=request.question,
            context_chunks=search_results["documents"][0],
            metadata=search_results["metadatas"][0] if search_results["metadatas"] else []
        )
        
        # Format sources
        sources = []
        if search_results["metadatas"]:
            for meta in search_results["metadatas"][0]:
                sources.append({
                    "heading": meta.get("heading", "Unknown"),
                    "document": meta.get("filename", "Unknown"),
                    "images": meta.get("images", "").split(";") if meta.get("images") else []
                })
        
        return QueryResponse(
            question=request.question,
            answer=response["answer"],
            sources=sources,
            confidence_score=response.get("confidence", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/documents/", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of all processed documents"""
    try:
        documents = await vector_store.list_documents()
        return [
            DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                upload_date=doc["upload_date"],
                chunk_count=doc["chunk_count"],
                status=doc["status"]
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a processed document and its data"""
    try:
        success = await vector_store.delete_document(document_id)
        if success:
            return JSONResponse(content={"message": "Document deleted successfully"})
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )