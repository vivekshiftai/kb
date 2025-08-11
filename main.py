"""
RAG PDF Processing API - Backend Only
A high-performance FastAPI application for PDF processing and intelligent querying
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import time
import logging
from typing import List
from datetime import datetime

from models.schemas import (
    QueryRequest, QueryResponse, PDFListResponse, PDFInfo,
    UploadResponse, HealthResponse, RulesRequest, RulesResponse
)
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from services.openai_client import OpenAIClient
from services.rules_generator import RulesGenerator
from config.settings import get_settings
from utils.file_utils import ensure_directories, get_file_hash
from utils.helpers import validate_pdf_file, clean_filename, format_file_size
import shutil

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

logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG PDF Processing API",
    description="Backend API for processing PDFs and querying content using Pinecone vector database",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize settings and services
settings = get_settings()
pdf_processor = PDFProcessor()
vector_store = VectorStore()
openai_client = OpenAIClient()
rules_generator = RulesGenerator()

# Ensure required directories exist
ensure_directories()

# Mount static files for serving extracted images
images_dir = os.path.join(settings.OUTPUT_DIR, "images")
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting RAG PDF Processing API...")
        await vector_store.initialize()
        logger.info("Application started successfully", 
                   version="2.0.0",
                   pinecone_index=settings.PINECONE_INDEX_NAME)
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


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with service information"""
    return {
        "service": "RAG PDF Processing API",
        "version": "2.0.0",
        "status": "running",
        "description": "Backend API for PDF processing and intelligent querying",
        "endpoints": {
            "health": "/health/",
            "upload": "/upload-pdf/",
            "list_pdfs": "/pdfs/",
            "query": "/query/",
            "rules": "/rules/",
            "delete": "/pdfs/{filename}",
            "docs": "/docs",
            "images": "/images/"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        start_time = time.time()
        
        # Check vector store connection
        vector_store_status = await vector_store.health_check()
        
        # Check OpenAI client connection
        openai_status = openai_client.check_connection()
        
        # Check file system
        upload_dir_exists = os.path.exists(settings.UPLOAD_DIR)
        output_dir_exists = os.path.exists(settings.OUTPUT_DIR)
        
        health_time = time.time() - start_time
        
        overall_status = "healthy" if all([
            vector_store_status, 
            openai_status, 
            upload_dir_exists, 
            output_dir_exists
        ]) else "degraded"
        
        logger.info("Health check completed", 
                   status=overall_status,
                   check_time=health_time)
        
        return HealthResponse(
            status=overall_status,
            message="All services operational" if overall_status == "healthy" else "Some services have issues",
            vector_store_status="connected" if vector_store_status else "error",
            openai_status="connected" if openai_status else "error",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/upload-pdf/", response_model=UploadResponse, tags=["PDF Management"])
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    """
    Upload and process a PDF file
    
    - **file**: PDF file to upload (max 50MB)
    - Returns upload confirmation and processing status
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed. Please upload a .pdf file."
            )
        
        # Clean and validate filename
        clean_name = clean_filename(file.filename)
        if not clean_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Read and validate file size
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
        
        # Validate PDF structure
        validation = validate_pdf_file(file_path)
        if not validation["valid"]:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {validation['error']}"
            )
        
        # Generate file hash for deduplication
        file_hash = get_file_hash(file_path)
        
        # Check if PDF already processed
        processed_pdfs = await vector_store.list_processed_pdfs()
        if clean_name in processed_pdfs:
            logger.info("PDF already processed", filename=clean_name)
            return UploadResponse(
                success=True,
                message="PDF already processed and available for querying",
                pdf_filename=clean_name,
                processing_status="completed"
            )
        
        # Start background processing
        background_tasks.add_task(
            process_pdf_background,
            file_path,
            clean_name,
            file_hash
        )
        
        upload_time = time.time() - start_time
        logger.info("PDF upload successful", 
                   filename=clean_name,
                   file_size=len(file_content),
                   upload_time=upload_time)
        
        return UploadResponse(
            success=True,
            message="PDF uploaded successfully. Processing started in background.",
            pdf_filename=clean_name,
            processing_status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading PDF", 
                    filename=getattr(file, 'filename', 'unknown'),
                    error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process PDF upload: {str(e)}"
        )


async def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary file {file_path}: {e}")

async def process_pdf_background(file_path: str, filename: str, file_hash: str):
    """Background task for PDF processing pipeline"""
    try:
        logger.info("Starting background PDF processing", filename=filename)
        processing_start = time.time()
        
        # Step 1: Process PDF (extract text, images, metadata)
        logger.info("Extracting content from PDF", filename=filename)
        processing_result = await pdf_processor.process_pdf(file_path, filename)
        
        # Step 2: Store in vector database
        logger.info("Storing chunks in vector database", 
                   filename=filename,
                   chunk_count=len(processing_result["chunks"]))
        
        document_id = await vector_store.store_document_chunks(
            pdf_filename=filename,
            chunks=processing_result["chunks"],
            file_hash=file_hash
        )
        
        processing_time = time.time() - processing_start
        
        logger.info("PDF processing pipeline completed", 
                   filename=filename,
                   document_id=document_id,
                   total_chunks=len(processing_result["chunks"]),
                   total_images=processing_result.get("total_images", 0),
                   processing_time=processing_time)
        
    except Exception as e:
        logger.error("Background PDF processing failed", 
                    filename=filename, 
                    error=str(e))


@app.get("/pdfs/", response_model=PDFListResponse, tags=["PDF Management"])
async def list_pdfs():
    """
    Get list of all processed PDF filenames with metadata
    
    Returns list of PDFs available for querying
    """
    try:
        start_time = time.time()
        
        # Get processed PDFs from vector store
        pdf_filenames = await vector_store.list_processed_pdfs()
        
        # Collect detailed information for each PDF
        pdf_infos = []
        for filename in pdf_filenames:
            try:
                stats = await vector_store.get_pdf_stats(filename)
                
                # Get file info if available
                file_path = os.path.join(settings.UPLOAD_DIR, filename)
                file_size = None
                upload_date = None
                
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    upload_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                pdf_infos.append(PDFInfo(
                    filename=filename,
                    chunk_count=stats.get("vector_count", 0),
                    file_size=file_size,
                    upload_date=upload_date
                ))
                
            except Exception as e:
                logger.warning("Error getting PDF stats", 
                             filename=filename, 
                             error=str(e))
                pdf_infos.append(PDFInfo(filename=filename))
        
        list_time = time.time() - start_time
        logger.info("PDF list retrieved", 
                   count=len(pdf_infos),
                   processing_time=list_time)
        
        return PDFListResponse(
            pdfs=pdf_infos,
            total_count=len(pdf_infos)
        )
        
    except Exception as e:
        logger.error("Error listing PDFs", error=str(e))
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve PDF list"
        )


@app.post("/query/", response_model=QueryResponse, tags=["Querying"])
async def query_pdf(request: QueryRequest):
    """
    Query a specific PDF with natural language
    
    - **pdf_filename**: Name of the PDF to query
    - **query**: Natural language question
    - **max_results**: Maximum number of relevant chunks to return (1-20)
    
    Returns AI-generated answer with supporting evidence and images
    """
    try:
        start_time = time.time()
        
        # Validate PDF exists
        processed_pdfs = await vector_store.list_processed_pdfs()
        if request.pdf_filename not in processed_pdfs:
            available_pdfs = ", ".join(processed_pdfs[:5])  # Show first 5
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{request.pdf_filename}' not found. Available PDFs: {available_pdfs}"
            )
        
        # Perform similarity search
        logger.info("Performing similarity search", 
                   pdf_filename=request.pdf_filename,
                   query=request.query[:100])
        
        search_results = await vector_store.search_pdf(
            pdf_filename=request.pdf_filename,
            query=request.query,
            top_k=request.max_results
        )
        
        # Handle no results case
        if not search_results["matches"]:
            no_results_time = time.time() - start_time
            logger.info("No relevant content found", 
                       pdf_filename=request.pdf_filename,
                       query=request.query[:50])
            
            return QueryResponse(
                pdf_filename=request.pdf_filename,
                query=request.query,
                answer="I couldn't find any relevant information for your query in this PDF. Try rephrasing your question or check if the content exists in the document.",
                results=[],
                total_matches=0,
                processing_time=no_results_time
            )
        
        # Generate AI response
        logger.info("Generating AI response", 
                   matches=len(search_results["matches"]))
        
        context_chunks = [match["text"] for match in search_results["matches"]]
        openai_response = await openai_client.generate_response(
            question=request.query,
            context_chunks=context_chunks,
            pdf_filename=request.pdf_filename
        )
        
        # Format results with accessible image URLs
        formatted_results = []
        for match in search_results["matches"]:
            # Process image paths to URLs
            image_infos = []
            for img_path in match.get("images", []):
                if os.path.exists(img_path):
                    # Create relative URL for static file serving
                    rel_path = os.path.relpath(img_path, images_dir)
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
                "page_number": match.get("page_number"),
                "images": image_infos
            })
        
        query_time = time.time() - start_time
        
        logger.info("Query completed successfully", 
                   pdf_filename=request.pdf_filename,
                   query_preview=request.query[:50],
                   matches=len(formatted_results),
                   processing_time=query_time)
        
        return QueryResponse(
            pdf_filename=request.pdf_filename,
            query=request.query,
            answer=openai_response["answer"],
            results=formatted_results,
            total_matches=search_results["total_matches"],
            processing_time=query_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing failed", 
                    pdf_filename=request.pdf_filename,
                    query=request.query[:50],
                    error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/rules/", response_model=RulesResponse, tags=["Rules Generation"])
async def generate_rules(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None, description="PDF file to analyze for rules generation"),
    pdf_filename: str = None,
    chunk_size: int = 10,
    rule_types: List[str] = ["monitoring", "maintenance", "alert"]
):
    """
    Generate IoT device rules and maintenance data from PDF content
    
    You can either:
    1. Upload a new PDF file directly (file parameter)
    2. Use an already uploaded PDF (pdf_filename parameter)
    
    - **file**: PDF file to upload and analyze (optional)
    - **pdf_filename**: Name of already uploaded PDF file to analyze (optional)
    - **chunk_size**: Number of pages to process in each chunk (default: 10)
    - **rule_types**: Types of rules to generate (default: ["monitoring", "maintenance", "alert"])
    
    Processes the PDF in chunks and generates:
    - IoT device monitoring and control rules
    - Maintenance schedules and requirements
    - Alert conditions and thresholds
    """
    try:
        # Validate input
        if not file and not pdf_filename:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'pdf_filename' must be provided"
            )
        
        if file and pdf_filename:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' or 'pdf_filename', not both"
            )
        
        if file:
            # Handle direct file upload
            logger.info("Processing uploaded PDF for rules generation", 
                       filename=file.filename,
                       chunk_size=chunk_size,
                       rule_types=rule_types)
            
            # Validate file
            if not validate_pdf_file(file):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file. Please upload a valid PDF."
                )
            
            # Save uploaded file temporarily
            temp_filename = f"temp_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(file.filename)}"
            temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Generate rules from the uploaded file
            result = await rules_generator.generate_rules_from_pdf(
                pdf_filename=temp_filename,
                chunk_size=chunk_size,
                rule_types=rule_types,
                pdf_file_path=temp_path
            )
            
            # Clean up temporary file
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
        else:
            # Use already uploaded PDF
            logger.info("Generating rules from existing PDF", 
                       filename=pdf_filename,
                       chunk_size=chunk_size,
                       rule_types=rule_types)
            
            # Check if PDF exists
            pdf_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
            if not os.path.exists(pdf_path):
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDF '{pdf_filename}' not found"
                )
            
            # Generate rules and maintenance data
            result = await rules_generator.generate_rules_from_pdf(
                pdf_filename=pdf_filename,
                chunk_size=chunk_size,
                rule_types=rule_types
            )
        
        logger.info("Rules generation completed successfully", 
                   filename=result["pdf_filename"],
                   rules_count=len(result["iot_rules"]),
                   maintenance_count=len(result["maintenance_data"]),
                   processing_time=result["processing_time"])
        
        return RulesResponse(
            pdf_filename=result["pdf_filename"],
            total_pages=result["total_pages"],
            processed_chunks=result["processed_chunks"],
            iot_rules=result["iot_rules"],
            maintenance_data=result["maintenance_data"],
            processing_time=result["processing_time"],
            summary=result["summary"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rules generation failed", 
                    filename=file.filename if file else pdf_filename,
                    error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate rules: {str(e)}"
        )


@app.delete("/pdfs/{pdf_filename}", tags=["PDF Management"])
async def delete_pdf(pdf_filename: str):
    """
    Delete a PDF and all associated data
    
    - **pdf_filename**: Name of the PDF to delete
    
    Removes PDF file, vector embeddings, and extracted images
    """
    try:
        logger.info("Deleting PDF", filename=pdf_filename)
        
        # Delete from vector store
        success = await vector_store.delete_pdf(pdf_filename)
        
        if success:
            # Delete uploaded file
            file_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Deleted PDF file", path=file_path)
            
            # Delete extracted images directory
            pdf_name = os.path.splitext(pdf_filename)[0]
            images_path = os.path.join(images_dir, pdf_name)
            if os.path.exists(images_path):
                import shutil
                shutil.rmtree(images_path)
                logger.info("Deleted images directory", path=images_path)
            
            logger.info("PDF deletion completed", filename=pdf_filename)
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"PDF '{pdf_filename}' and all associated data deleted successfully"
                }
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{pdf_filename}' not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF deletion failed", 
                    filename=pdf_filename, 
                    error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete PDF: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )