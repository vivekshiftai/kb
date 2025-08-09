from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import time
import logging
from typing import List
from datetime import datetime

from models.schemas import (
    QueryRequest, QueryResponse, PDFListResponse, PDFInfo,
    UploadResponse, HealthResponse, ErrorResponse
)
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.openai_client import OpenAIClient
from config.settings import get_settings
from utils.file_utils import ensure_directories, get_file_hash
from utils.helpers import validate_pdf_file, clean_filename, format_file_size

# Configure structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Processing API",
    description="API for processing PDFs and querying content using Pinecone vector database",
    version="2.0.0"
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

# Mount static files for serving images
if not os.path.exists(os.path.join(settings.OUTPUT_DIR, "images")):
    os.makedirs(os.path.join(settings.OUTPUT_DIR, "images"), exist_ok=True)

app.mount("/images", StaticFiles(directory=os.path.join(settings.OUTPUT_DIR, "images")), name="images")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await vector_store.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

@app.get("/health/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        vector_store_status = await vector_store.health_check()
        
        # Check OpenAI client
        openai_status = openai_client.check_connection()
        
        return HealthResponse(
            status="healthy" if vector_store_status and openai_status else "degraded",
            message="All services are running" if vector_store_status and openai_status else "Some services have issues",
            vector_store_status="connected" if vector_store_status else "error",
            openai_status="connected" if openai_status else "error",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a PDF file"""
    
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Clean filename
        clean_name = clean_filename(file.filename)
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {format_file_size(settings.MAX_FILE_SIZE)}"
            )
        
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIR, clean_name)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Validate PDF
        validation = validate_pdf_file(file_path)
        if not validation["valid"]:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {validation['error']}")
        
        # Get file hash for deduplication
        file_hash = get_file_hash(file_path)
        
        # Check if already processed
        processed_pdfs = await vector_store.list_processed_pdfs()
        if clean_name in processed_pdfs:
            logger.info("PDF already processed", filename=clean_name)
            return UploadResponse(
                success=True,
                message="PDF already processed",
                pdf_filename=clean_name,
                processing_status="completed"
            )
        
        # Process PDF in background
        background_tasks.add_task(
            process_pdf_background,
            file_path,
            clean_name,
            file_hash
        )
        
        processing_time = time.time() - start_time
        logger.info("PDF upload successful", 
                   filename=clean_name, 
                   processing_time=processing_time)
        
        return UploadResponse(
            success=True,
            message="PDF uploaded successfully, processing started",
            pdf_filename=clean_name,
            processing_status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading PDF", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

async def process_pdf_background(file_path: str, filename: str, file_hash: str):
    """Background task to process PDF"""
    try:
        logger.info("Starting PDF processing", filename=filename)
        
        # Process PDF
        processing_result = await pdf_processor.process_pdf(file_path, filename)
        
        # Store in vector database
        document_id = await vector_store.store_document_chunks(
            pdf_filename=filename,
            chunks=processing_result["chunks"],
            file_hash=file_hash
        )
        
        logger.info("PDF processing completed", 
                   filename=filename, 
                   document_id=document_id,
                   chunks=len(processing_result["chunks"]))
        
    except Exception as e:
        logger.error("Error in background PDF processing", 
                    filename=filename, 
                    error=str(e))

@app.get("/pdfs/", response_model=PDFListResponse)
async def list_pdfs():
    """Get list of all processed PDF filenames"""
    try:
        start_time = time.time()
        
        # Get list from vector store
        pdf_filenames = await vector_store.list_processed_pdfs()
        
        # Get additional info for each PDF
        pdf_infos = []
        for filename in pdf_filenames:
            try:
                stats = await vector_store.get_pdf_stats(filename)
                pdf_infos.append(PDFInfo(
                    filename=filename,
                    chunk_count=stats.get("vector_count", 0)
                ))
            except Exception as e:
                logger.warning("Error getting PDF stats", filename=filename, error=str(e))
                pdf_infos.append(PDFInfo(filename=filename))
        
        processing_time = time.time() - start_time
        logger.info("Listed PDFs", count=len(pdf_infos), processing_time=processing_time)
        
        return PDFListResponse(
            pdfs=pdf_infos,
            total_count=len(pdf_infos)
        )
        
    except Exception as e:
        logger.error("Error listing PDFs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve PDF list")

@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """Query a specific PDF with a question"""
    try:
        start_time = time.time()
        
        # Validate PDF exists
        processed_pdfs = await vector_store.list_processed_pdfs()
        if request.pdf_filename not in processed_pdfs:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{request.pdf_filename}' not found. Available PDFs: {processed_pdfs}"
            )
        
        # Search vector database
        search_results = await vector_store.search_pdf(
            pdf_filename=request.pdf_filename,
            query=request.query,
            top_k=request.max_results
        )
        
        if not search_results["matches"]:
            processing_time = time.time() - start_time
            return QueryResponse(
                pdf_filename=request.pdf_filename,
                query=request.query,
                answer="I couldn't find any relevant information for your query in this PDF.",
                results=[],
                total_matches=0,
                processing_time=processing_time
            )
        
        # Generate answer using OpenAI
        context_chunks = [match["text"] for match in search_results["matches"]]
        openai_response = await openai_client.generate_response(
            question=request.query,
            context_chunks=context_chunks,
            pdf_filename=request.pdf_filename
        )
        
        # Format results with image URLs
        formatted_results = []
        for match in search_results["matches"]:
            # Convert image paths to URLs
            image_infos = []
            for img_path in match.get("images", []):
                if os.path.exists(img_path):
                    # Create relative URL for the image
                    rel_path = os.path.relpath(img_path, os.path.join(settings.OUTPUT_DIR, "images"))
                    image_url = f"/images/{rel_path}"
                    
                    image_infos.append({
                        "filename": os.path.basename(img_path),
                        "url": image_url,
                        "page_number": match.get("page_number", 1)
                    })
            
            formatted_results.append({
                "heading": match["heading"],
                "text": match["text"],
                "score": match["score"],
                "images": image_infos
            })
        
        processing_time = time.time() - start_time
        
        logger.info("Query processed", 
                   pdf_filename=request.pdf_filename,
                   query=request.query[:50],
                   matches=len(formatted_results),
                   processing_time=processing_time)
        
        return QueryResponse(
            pdf_filename=request.pdf_filename,
            query=request.query,
            answer=openai_response["answer"],
            results=formatted_results,
            total_matches=search_results["total_matches"],
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing query", 
                    pdf_filename=request.pdf_filename,
                    query=request.query,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.delete("/pdfs/{pdf_filename}")
async def delete_pdf(pdf_filename: str):
    """Delete a PDF and all its associated data"""
    try:
        success = await vector_store.delete_pdf(pdf_filename)
        
        if success:
            # Also delete the uploaded file if it exists
            file_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info("PDF deleted", filename=pdf_filename)
            return JSONResponse(content={"message": f"PDF '{pdf_filename}' deleted successfully"})
        else:
            raise HTTPException(status_code=404, detail="PDF not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting PDF", filename=pdf_filename, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete PDF")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG PDF Processing API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "/upload-pdf/",
            "list": "/pdfs/",
            "query": "/query/",
            "delete": "/pdfs/{filename}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )