"""
RAG PDF Processing API - Backend Only
A high-performance FastAPI application for PDF processing and intelligent querying
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys
import time
import logging
import chromadb
from typing import List
from datetime import datetime

from models.schemas import (
    QueryRequest, QueryResponse, PDFListResponse, PDFInfo,
    UploadResponse, HealthResponse, RulesRequest, RulesResponse, SafetyPrecaution,
    ImageInfo, QueryResult
)
from services.pdf_processor import PDFProcessor
from services.openai_client import OpenAIClient
from services.rules_generator import RulesGenerator
from services.minieu_processor import MinieuProcessor
from services.chromadb_manager import ChromaDBManager
from config.settings import get_settings
from utils.file_utils import ensure_directories, get_file_hash
from utils.helpers import validate_pdf_file, clean_filename, format_file_size
import shutil

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

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
openai_client = OpenAIClient()
rules_generator = RulesGenerator()
minieu_processor = MinieuProcessor()
chromadb_manager = ChromaDBManager()

# Ensure required directories exist
ensure_directories()

# Mount static files for serving extracted images from Minieu output
images_dir = settings.MINIEU_OUTPUT_DIR
os.makedirs(images_dir, exist_ok=True)
logger.info(f"Mounting Minieu output directory for images", images_dir=images_dir, exists=os.path.exists(images_dir))

# Check if Minieu output directory has any image files
if os.path.exists(images_dir):
    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    logger.info(f"Found {len(image_files)} image files in Minieu output directory")

app.mount("/images", StaticFiles(directory=images_dir), name="images")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("🚀 Starting RAG PDF Processing API...")
        logger.info("📋 Initializing services...")
        
        logger.info("🔧 Checking OpenAI client...")
        openai_status = openai_client.check_connection()
        logger.info(f"✅ OpenAI client status: {openai_status}")
        
        logger.info("🔧 Checking directories...")
        ensure_directories()
        logger.info("✅ Directories verified")
        
        logger.info("🔧 Checking ChromaDB availability...")
        try:
            import chromadb
            test_client = chromadb.PersistentClient(path="./test_chromadb")
            test_client.heartbeat()
            logger.info("✅ ChromaDB is available")
        except Exception as e:
            logger.warning("⚠️ ChromaDB health check failed", error=str(e))
        
        logger.info("🔧 Checking Minieu availability...")
        minieu_available = minieu_processor.check_minieu_availability()
        if not minieu_available:
            logger.warning("⚠️ Minieu is not available. PDF processing will fail.")
        else:
            logger.info("✅ Minieu is available")
        
        logger.info("🎉 Application started successfully", 
                   version="2.0.0",
                   upload_dir=settings.UPLOAD_DIR,
                   output_dir=settings.OUTPUT_DIR,
                   minieu_output_dir=settings.MINIEU_OUTPUT_DIR)
    except Exception as e:
        logger.error("❌ Failed to start application", error=str(e))
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
            "images": "/images/",
            "debug": {
                "minieu_status": "/debug/minieu-status/",
                "process_minieu": "/debug/process-with-minieu/",
                "query_test": "/debug/query-test/",
                "images": "/debug/images/"
            }
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/debug/images/", tags=["Debug"])
async def debug_images():
    """Debug endpoint to check image directory and files"""
    try:
        # Check if Minieu output directory exists
        images_exist = os.path.exists(images_dir)
        images_list = []
        
        if images_exist:
            # List all image files in Minieu output directory
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        rel_path = os.path.relpath(os.path.join(root, file), images_dir)
                        images_list.append({
                            "filename": file,
                            "relative_path": rel_path,
                            "full_path": os.path.join(root, file),
                            "size": os.path.getsize(os.path.join(root, file)),
                            "url": f"/images/{rel_path}"
                        })
        
        return {
            "minieu_output_directory": images_dir,
            "directory_exists": images_exist,
            "total_images": len(images_list),
            "images": images_list[:10]  # Return first 10 images
        }
    except Exception as e:
        logger.error("Error in debug images endpoint", error=str(e))
        return {"error": str(e)}


@app.get("/debug/minieu-status/", tags=["Debug"])
async def debug_minieu_status():
    """Debug endpoint to check Minieu output status and structure"""
    try:
        minieu_dir = settings.MINIEU_OUTPUT_DIR
        status = {
            "minieu_output_dir": minieu_dir,
            "exists": os.path.exists(minieu_dir),
            "minieu_available": minieu_processor.check_minieu_availability(),
            "minieu_version": minieu_processor.get_minieu_version(),
            "pdfs_processed": []
        }
        
        if os.path.exists(minieu_dir):
            for item in os.listdir(minieu_dir):
                item_path = os.path.join(minieu_dir, item)
                if os.path.isdir(item_path):
                    # Check for auto directory
                    auto_found = False
                    auto_path = None
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            auto_check = os.path.join(subitem_path, "auto")
                            if os.path.exists(auto_check):
                                auto_found = True
                                auto_path = auto_check
                                break
                    
                    status["pdfs_processed"].append({
                        "pdf_name": item,
                        "pdf_filename": f"{item}.pdf",
                        "has_auto_dir": auto_found,
                        "auto_path": auto_path,
                        "markdown_files": [],
                        "image_files": []
                    })
                    
                    if auto_found and auto_path:
                        # Count markdown files
                        md_files = [f for f in os.listdir(auto_path) if f.endswith('.md')]
                        status["pdfs_processed"][-1]["markdown_files"] = md_files
                        
                        # Count images
                        images_dir = os.path.join(auto_path, "images")
                        if os.path.exists(images_dir):
                            img_files = [f for f in os.listdir(images_dir) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                            status["pdfs_processed"][-1]["image_files"] = img_files
        
        # Add validation summary
        status["validation"] = {
            "minieu_dir_exists": os.path.exists(minieu_dir),
            "minieu_dir_writable": os.access(minieu_dir, os.W_OK) if os.path.exists(minieu_dir) else False,
            "total_pdfs_found": len(status["pdfs_processed"]),
            "pdfs_with_auto_dir": len([p for p in status["pdfs_processed"] if p["has_auto_dir"]]),
            "total_markdown_files": sum(len(p["markdown_files"]) for p in status["pdfs_processed"]),
            "total_image_files": sum(len(p["image_files"]) for p in status["pdfs_processed"])
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking Minieu status: {e}")
        return {"error": str(e)}

@app.post("/debug/process-with-minieu/", tags=["Debug"])
async def debug_process_with_minieu(pdf_filename: str):
    """Debug endpoint to manually trigger Minieu processing for a PDF"""
    try:
        logger.info(f"🔧 Manual Minieu processing requested for: {pdf_filename}")
        
        # Check if PDF exists
        pdf_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{pdf_filename}' not found in uploads directory"
            )
        
        # Check if Minieu is available
        if not minieu_processor.check_minieu_availability():
            raise HTTPException(
                status_code=503,
                detail="Minieu is not available. Please install Minieu first."
            )
        
        # Process with Minieu
        result = await minieu_processor.process_pdf_with_minieu(pdf_path, pdf_filename)
        
        return {
            "success": True,
            "message": "Minieu processing completed",
            "pdf_filename": pdf_filename,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual Minieu processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/query-test/", tags=["Debug"])
async def debug_query_test():
    """Debug endpoint to test query functionality with a simple search"""
    try:
        # Get list of processed PDFs from Minieu output
        processed_pdfs = []
        if os.path.exists(settings.MINIEU_OUTPUT_DIR):
            for item in os.listdir(settings.MINIEU_OUTPUT_DIR):
                item_path = os.path.join(settings.MINIEU_OUTPUT_DIR, item)
                if os.path.isdir(item_path):
                    # Look for subdirectory with 'auto'
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            auto_path = os.path.join(subitem_path, "auto")
                            if os.path.exists(auto_path):
                                processed_pdfs.append(f"{item}.pdf")
                                break
        
        if not processed_pdfs:
            return {"error": "No processed PDFs found in Minieu output"}
        
        # Test with first PDF
        test_pdf = processed_pdfs[0]
        pdf_name = os.path.splitext(test_pdf)[0]
        
        # Initialize ChromaDB client
        chromadb_path = f"./chroma_db_{pdf_name}"
        if not os.path.exists(chromadb_path):
            return {"error": f"ChromaDB not found for {test_pdf}"}
        
        client = chromadb.PersistentClient(path=chromadb_path)
        try:
            md_collection = client.get_collection("md_heading_chunks")
        except Exception as e:
            logger.warning(f"⚠️ Primary ChromaDB collection retrieval failed, trying v1.0.x format: {e}")
            try:
                # Fallback for ChromaDB v1.0.x
                md_collection = client.get_collection(
                    name="md_heading_chunks"
                )
            except Exception as e2:
                logger.error(f"❌ ChromaDB collection retrieval failed with both methods: {e2}")
                raise HTTPException(status_code=500, detail=f"ChromaDB collection retrieval failed: {e2}")
        
        # Perform a simple search
        search_results = md_collection.query(
            query_texts=["test"],
            n_results=3,
            include=["metadatas", "documents"]
        )
        
        # Check for images in results
        matches_with_images = []
        for i, meta in enumerate(search_results["metadatas"][0]):
            images = meta.get("images", "")
            if images:
                img_list = [img.strip() for img in images.split(";") if img.strip()]
                if img_list:
                    matches_with_images.append({
                        "match_index": i,
                        "heading": meta.get("heading", ""),
                        "image_count": len(img_list),
                        "images": img_list[:3]  # Show first 3 image paths
                    })
        
        return {
            "test_pdf": test_pdf,
            "total_matches": len(search_results["documents"][0]),
            "matches_with_images": len(matches_with_images),
            "matches_with_images_details": matches_with_images
        }
        
    except Exception as e:
        logger.error("Error in debug query test", error=str(e))
        return {"error": str(e)}


@app.get("/health/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        start_time = time.time()
        
        # Check OpenAI client connection
        openai_status = openai_client.check_connection()
        
        # Check file system
        upload_dir_exists = os.path.exists(settings.UPLOAD_DIR)
        output_dir_exists = os.path.exists(settings.OUTPUT_DIR)
        minieu_output_dir_exists = os.path.exists(settings.MINIEU_OUTPUT_DIR)
        
        # Check ChromaDB availability
        chromadb_status = True
        try:
            import chromadb
            # Test ChromaDB connection
            test_client = chromadb.PersistentClient(path="./test_chromadb")
            test_client.heartbeat()
            chromadb_status = True
        except Exception as e:
            logger.warning("ChromaDB health check failed", error=str(e))
            chromadb_status = False
        
        # Check Minieu availability and version
        minieu_status = minieu_processor.check_minieu_availability()
        minieu_version = minieu_processor.get_minieu_version()
        
        health_time = time.time() - start_time
        
        overall_status = "healthy" if all([
            openai_status, 
            upload_dir_exists, 
            output_dir_exists,
            minieu_output_dir_exists,
            chromadb_status,
            minieu_status
        ]) else "degraded"
        
        logger.info("Health check completed", 
                   status=overall_status,
                   check_time=health_time)
        
        return HealthResponse(
            status=overall_status,
            message="All services operational" if overall_status == "healthy" else "Some services have issues",
            vector_store_status="connected" if chromadb_status else "error",
            openai_status="connected" if openai_status else "error",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/upload-pdf/", response_model=UploadResponse, tags=["PDF Management"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload and process")
):
    """
    Upload and process a PDF file
    
    - **file**: PDF file to upload (max 50MB)
    - Returns upload confirmation and processing status
    """
    start_time = time.time()
    
    try:
        logger.info("📤 Starting PDF upload process", filename=file.filename)
        
        # Validate file type
        logger.info("🔍 Validating file type...")
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            logger.warning("❌ Invalid file type uploaded", filename=file.filename)
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed. Please upload a .pdf file."
            )
        logger.info("✅ File type validation passed")
        
        # Clean and validate filename
        logger.info("🔍 Cleaning filename...")
        clean_name = clean_filename(file.filename)
        if not clean_name:
            logger.warning("❌ Invalid filename", original_name=file.filename)
            raise HTTPException(status_code=400, detail="Invalid filename")
        logger.info("✅ Filename cleaned", original_name=file.filename, clean_name=clean_name)
        
        # Read and validate file size
        logger.info("📏 Reading file content...")
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"📊 File size: {format_file_size(file_size)}")
        
        if file_size > settings.MAX_FILE_SIZE:
            logger.warning("❌ File too large", 
                          file_size=format_file_size(file_size),
                          max_size=format_file_size(settings.MAX_FILE_SIZE))
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {format_file_size(settings.MAX_FILE_SIZE)}"
            )
        logger.info("✅ File size validation passed")
        
        # Save uploaded file
        logger.info("💾 Saving file to disk...")
        file_path = os.path.join(settings.UPLOAD_DIR, clean_name)
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info("✅ File saved successfully", path=file_path)
        
        # Validate PDF structure
        logger.info("🔍 Validating PDF structure...")
        validation = validate_pdf_file(file_path)
        if not validation["valid"]:
            logger.error("❌ PDF validation failed", error=validation['error'])
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid PDF file: {validation['error']}"
            )
        logger.info("✅ PDF structure validation passed")
        
        # Generate file hash for deduplication
        logger.info("🔐 Generating file hash...")
        file_hash = get_file_hash(file_path)
        logger.info("✅ File hash generated", hash=file_hash[:8] + "...")
        
        # Check if PDF already processed in Minieu output and ChromaDB
        logger.info("🔍 Checking if PDF already processed...")
        pdf_name = os.path.splitext(clean_name)[0]
        minieu_output_dir = os.path.join(settings.MINIEU_OUTPUT_DIR, pdf_name)
        
        # Check Minieu output
        minieu_processed = False
        if os.path.exists(minieu_output_dir):
            # Check if auto directory exists
            auto_dir = None
            for item in os.listdir(minieu_output_dir):
                item_path = os.path.join(minieu_output_dir, item)
                if os.path.isdir(item_path):
                    # Look for subdirectory with 'auto'
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            auto_path = os.path.join(subitem_path, "auto")
                            if os.path.exists(auto_path):
                                auto_dir = auto_path
                                break
                    if auto_dir:
                        break
            
            if auto_dir:
                minieu_processed = True
                logger.info("✅ PDF already processed in Minieu output", filename=clean_name)
        
        # Check ChromaDB
        chromadb_processed = chromadb_manager.collection_exists(pdf_name)
        if chromadb_processed:
            logger.info("✅ PDF already processed in ChromaDB", filename=clean_name)
        
        # If both Minieu and ChromaDB are processed, return success
        if minieu_processed and chromadb_processed:
            return UploadResponse(
                success=True,
                message="PDF already processed and available for querying",
                pdf_filename=clean_name,
                processing_status="completed"
            )
        
        logger.info("✅ PDF not previously processed")
        
        # Check if Minieu output exists for this PDF
        if not os.path.exists(minieu_output_dir):
            logger.warning(f"⚠️ Minieu output not found for {clean_name}")
            logger.info(f"📋 The PDF will be uploaded but processing will fail until Minieu output is available")
            logger.info(f"📋 Expected Minieu output location: {minieu_output_dir}")
        
        # Process PDF immediately (no background task)
        logger.info("🚀 Starting immediate PDF processing...")
        try:
            processing_result = await process_pdf_background(file_path, clean_name, file_hash)
            
            upload_time = time.time() - start_time
            logger.info("🎉 PDF upload and processing completed successfully", 
                       filename=clean_name,
                       file_size=format_file_size(file_size),
                       upload_time=f"{upload_time:.2f}s")
            
            return UploadResponse(
                success=True,
                message="PDF uploaded and processed successfully.",
                pdf_filename=clean_name,
                processing_status="completed"
            )
        except Exception as processing_error:
            logger.error("❌ Error processing PDF", 
                        filename=clean_name,
                        error=str(processing_error))
            
            # Clean up the uploaded file if processing failed
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🧹 Cleaned up failed upload: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up file: {cleanup_error}")
            
            upload_time = time.time() - start_time
            return UploadResponse(
                success=False,
                message=f"PDF uploaded but processing failed: {str(processing_error)}",
                pdf_filename=clean_name,
                processing_status="processing_failed"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Error uploading PDF", 
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
    """Background task for PDF processing pipeline using Minieu output"""
    try:
        logger.info("🔄 Starting background PDF processing", 
                   filename=filename,
                   file_path=file_path,
                   file_hash=file_hash[:8] + "...")
        processing_start = time.time()
        
        # Step 1: Process PDF with Minieu
        logger.info("🤖 Step 1: Processing PDF with Minieu", 
                   filename=filename,
                   step="minieu_processing")
        minieu_result = await minieu_processor.process_pdf_with_minieu(file_path, filename)
        
        logger.info("✅ Minieu processing completed", 
                   filename=filename,
                   minieu_time=minieu_result.get("processing_time", 0),
                   step="minieu_complete")
        
        # Step 2: Process PDF using Minieu output data
        logger.info("📄 Step 2: Processing PDF using Minieu output data", 
                   filename=filename,
                   step="content_extraction")
        processing_result = await pdf_processor.process_pdf(file_path, filename)
        
        logger.info("✅ PDF content extraction completed", 
                   filename=filename,
                   chunks=len(processing_result["chunks"]),
                   images=processing_result.get("total_images", 0),
                   step="content_extraction_complete")
        
        # Step 2: Store in ChromaDB
        logger.info("🗄️ Step 2: Storing chunks in ChromaDB", 
                   filename=filename,
                   chunk_count=len(processing_result["chunks"]),
                   step="vector_storage")
        
        # Store chunks using ChromaDB manager
        pdf_name = os.path.splitext(filename)[0]
        storage_success = chromadb_manager.store_chunks(pdf_name, processing_result["chunks"])
        
        if not storage_success:
            raise Exception("Failed to store chunks in ChromaDB")
        
        logger.info("✅ ChromaDB storage completed", 
                   filename=filename,
                   total_chunks=len(processing_result["chunks"]),
                   step="vector_storage_complete")
        
        processing_time = time.time() - processing_start
        
        logger.info("🎉 PDF processing pipeline completed successfully", 
                   filename=filename,
                   total_chunks=len(processing_result["chunks"]),
                   total_images=processing_result.get("total_images", 0),
                   processing_time=f"{processing_time:.2f}s",
                   step="pipeline_complete")
        
    except Exception as e:
        logger.error("❌ Background PDF processing failed", 
                    filename=filename, 
                    error=str(e),
                    step="pipeline_failed")
        import traceback
        logger.error("❌ Full error traceback:", 
                    traceback=traceback.format_exc(),
                    filename=filename)


@app.get("/pdfs/", response_model=PDFListResponse, tags=["PDF Management"])
async def list_pdfs():
    """
    Get list of all processed PDF filenames with metadata
    
    Returns list of PDFs available for querying
    """
    try:
        start_time = time.time()
        logger.info("📋 Starting PDF list retrieval", step="list_start")
        
        # Get processed PDFs from ChromaDB collections
        logger.info("🔍 Scanning ChromaDB collections for processed PDFs", step="chromadb_scan")
        pdf_infos = []
        
        # Get all PDF collections from ChromaDB
        pdf_collections = chromadb_manager.list_collections()
        
        for collection_name in pdf_collections:
            # Extract PDF name from collection name (format: pdf_<pdf_name>)
            pdf_name = collection_name.replace("pdf_", "")
            pdf_filename = f"{pdf_name}.pdf"
            
            # Get collection stats
            stats = chromadb_manager.get_collection_stats(pdf_name)
            
            # Check if Minieu output exists
            minieu_output_dir = os.path.join(settings.MINIEU_OUTPUT_DIR, pdf_name)
            minieu_processed = os.path.exists(minieu_output_dir)
            
            # Check if auto directory exists
            auto_found = False
            if minieu_processed:
                for item in os.listdir(minieu_output_dir):
                    item_path = os.path.join(minieu_output_dir, item)
                    if os.path.isdir(item_path):
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path):
                                auto_check = os.path.join(subitem_path, "auto")
                                if os.path.exists(auto_check):
                                    auto_found = True
                                    break
            # Create PDF info
            pdf_infos.append(PDFInfo(
                filename=pdf_filename,
                chunk_count=stats.get("count", 0),
                file_size=None,  # We don't store file size in ChromaDB
                upload_date=None  # We don't store upload date in ChromaDB
            ))
            
            logger.info(f"📄 Found processed PDF: {pdf_filename}", 
                       chunk_count=stats.get("count", 0),
                       minieu_processed=minieu_processed,
                       auto_found=auto_found,
                       step="pdf_found")
        
        list_time = time.time() - start_time
        logger.info("✅ PDF list retrieval completed", 
                   count=len(pdf_infos),
                   processing_time=list_time,
                   step="list_complete")
        
        return PDFListResponse(
            pdfs=pdf_infos,
            total_count=len(pdf_infos)
        )
        
    except Exception as e:
        logger.error("❌ Error listing PDFs", 
                    error=str(e),
                    step="list_error")
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve PDF list"
        )


@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest):
    """Query a specific PDF using Minieu-processed content"""
    try:
        query_start = time.time()
        
        logger.info(f"🔍 Starting query for PDF: {request.pdf_filename}", 
                   query=request.query,
                   max_results=request.max_results,
                   step="query_start")
        
        # Validate PDF exists in Minieu output
        pdf_name = os.path.splitext(request.pdf_filename)[0]
        minieu_output_dir = os.path.join(settings.MINIEU_OUTPUT_DIR, pdf_name)
        
        if not os.path.exists(minieu_output_dir):
            raise HTTPException(
                status_code=404, 
                detail=f"PDF '{request.pdf_filename}' not found in Minieu output directory"
            )
        
        # Find auto directory
        auto_dir = None
        for item in os.listdir(minieu_output_dir):
            item_path = os.path.join(minieu_output_dir, item)
            if os.path.isdir(item_path):
                # Look for subdirectory with 'auto'
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        auto_path = os.path.join(subitem_path, "auto")
                        if os.path.exists(auto_path):
                            auto_dir = auto_path
                            break
                if auto_dir:
                    break
        
        if not auto_dir:
            raise HTTPException(
                status_code=404,
                detail=f"No 'auto' directory found for PDF '{request.pdf_filename}'"
            )
        
        logger.info(f"📁 Found auto directory: {auto_dir}", 
                   step="auto_dir_found")
        
        # Perform semantic search using ChromaDB manager
        logger.info("🔍 Performing semantic search on markdown chunks...", 
                   step="semantic_search")
        
        search_results = chromadb_manager.search(pdf_name, request.query, request.max_results)
        
        if not search_results["documents"][0]:
            logger.warning("No relevant chunks found for query", 
                          query=request.query)
            return QueryResponse(
                pdf_filename=request.pdf_filename,
                query=request.query,
                answer="No relevant information found for your query.",
                results=[],
                total_matches=0,
                processing_time=time.time() - query_start
            )
        
        # Process search results
        results = []
        for idx, (doc, meta) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
            # Extract images from metadata
            images = []
            img_paths = [img.strip() for img in meta.get("images", "").split(";") if img.strip()]
            
            for img_path in img_paths:
                # Try to resolve image path with better logic
                possible_paths = [
                    img_path,  # Direct path
                    os.path.join(auto_dir, "images", img_path),  # In images subdirectory
                    os.path.join(auto_dir, img_path),  # In auto directory
                    os.path.join(settings.MINIEU_OUTPUT_DIR, pdf_name, "images", img_path),  # Alternative location
                ]
                
                abs_img_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        abs_img_path = path
                        break
                
                if abs_img_path:
                    try:
                        # Create URL for the image - use Minieu output directory as base
                        rel_path = os.path.relpath(abs_img_path, settings.MINIEU_OUTPUT_DIR)
                        image_url = f"/images/{rel_path.replace(os.sep, '/')}"
                        
                        images.append(ImageInfo(
                            filename=os.path.basename(img_path),
                            url=image_url,
                            page_number=meta.get("page_number", 1)
                        ))
                        
                        logger.debug(f"✅ Image path resolved successfully", 
                                   original_path=img_path,
                                   resolved_path=abs_img_path,
                                   url=image_url)
                    except ValueError as e:
                        logger.warning(f"⚠️ Could not create relative path for image", 
                                     img_path=img_path,
                                     resolved_path=abs_img_path,
                                     error=str(e))
                else:
                    logger.warning(f"⚠️ Image file not found", 
                                 img_path=img_path,
                                 possible_paths=possible_paths)
            
            # Create result
            result = QueryResult(
                heading=meta.get("heading", "Untitled"),
                text=doc[:800] + ("..." if len(doc) > 800 else ""),
                score=search_results.get("distances", [[]])[0][idx] if search_results.get("distances") else 0.0,
                page_number=meta.get("page_number", 1),
                images=images
            )
            results.append(result)
        
        # Generate AI response using the best chunk
        best_chunk = search_results["documents"][0][0]  # Use the first (best) result
        best_meta = search_results["metadatas"][0][0]
        
        logger.info("🤖 Generating AI response...", 
                   step="ai_response")
        
        # Create context from top chunks
        context_chunks = []
        for doc, meta in zip(search_results["documents"][0][:3], search_results["metadatas"][0][:3]):
            context_chunks.append({
                "heading": meta.get("heading", "Untitled"),
                "text": doc,
                "images": meta.get("images", "")
            })
        
        # Generate response using OpenAI
        ai_response = await openai_client.generate_response_with_chunk_identification(
            question=request.query,
            context_chunks=context_chunks,
            pdf_filename=request.pdf_filename
        )
        
        # Filter results based on used chunks
        used_chunk_indices = ai_response.get("used_chunk_indices", [])
        if used_chunk_indices:
            used_matches = [results[i] for i in used_chunk_indices if i < len(results)]
        else:
            used_matches = results
        
        processing_time = time.time() - query_start
        
        logger.info(f"✅ Query completed successfully", 
                   pdf_filename=request.pdf_filename,
                   total_chunks_retrieved=len(results),
                   chunks_used=len(used_matches),
                   processing_time=f"{processing_time:.2f}s",
                   step="query_complete")
        
        return QueryResponse(
            pdf_filename=request.pdf_filename,
            query=request.query,
            answer=ai_response["answer"],
            results=used_matches,
            total_matches=len(used_matches),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error in query endpoint", 
                    pdf_filename=request.pdf_filename,
                    query=request.query,
                    error=str(e),
                    step="query_error")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/rules/", response_model=RulesResponse, tags=["Rules Generation"])
async def generate_rules(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None, description="PDF file to analyze for rules generation"),
    pdf_filename: str = None,
    chunk_size: int = 30,
    rule_types: str = Query("monitoring,maintenance,alert", description="Comma-separated rule types to generate")
):
    """
    Generate IoT device rules and maintenance data from PDF content
    
    You can either:
    1. Upload a new PDF file directly (file parameter)
    2. Use an already uploaded PDF (pdf_filename parameter)
    
    - **file**: PDF file to upload and analyze (optional)
    - **pdf_filename**: Name of already uploaded PDF file to analyze (optional)
    - **chunk_size**: Number of chunks to process in each batch (default: 30)
    - **rule_types**: Comma-separated rule types to generate (default: "monitoring,maintenance,alert")
    
    Processes the PDF using vector database chunks and generates:
    - IoT device monitoring and control rules
    - Maintenance schedules and requirements
    - Alert conditions and thresholds
    """
    try:
        logger.info("🤖 Starting IoT rules generation process")
        
        # Validate input
        logger.info("🔍 Validating input parameters...")
        if not file and not pdf_filename:
            logger.error("❌ No file or pdf_filename provided")
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'pdf_filename' must be provided"
            )
        
        if file and pdf_filename:
            logger.error("❌ Both file and pdf_filename provided")
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' or 'pdf_filename', not both"
            )
        logger.info("✅ Input validation passed")
        
        # Parse rule_types from comma-separated string
        logger.info("🔧 Parsing rule types...")
        rule_types_list = [rt.strip() for rt in rule_types.split(",") if rt.strip()]
        logger.info(f"📋 Rule types to generate: {rule_types_list}")
        
        if file:
            # Handle direct file upload
            logger.info("📤 Processing uploaded PDF for rules generation", 
                       filename=file.filename,
                       chunk_size=chunk_size,
                       rule_types=rule_types_list)
            
            # Validate file
            logger.info("🔍 Validating uploaded file...")
            if not validate_pdf_file(file):
                logger.error("❌ Invalid PDF file uploaded", filename=file.filename)
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file. Please upload a valid PDF."
                )
            logger.info("✅ File validation passed")
            
            # Save uploaded file temporarily
            logger.info("💾 Saving uploaded file temporarily...")
            temp_filename = f"temp_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(file.filename)}"
            temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info("✅ Temporary file saved", path=temp_path)
            
            # Generate rules from the uploaded file
            logger.info("🚀 Starting rules generation from uploaded file...")
            result = await rules_generator.generate_rules_from_pdf(
                pdf_filename=temp_filename,
                chunk_size=chunk_size,
                rule_types=rule_types_list,
                pdf_file_path=temp_path
            )
            
            # Clean up temporary file
            logger.info("🧹 Scheduling temporary file cleanup...")
            background_tasks.add_task(cleanup_temp_file, temp_path)
            
        else:
            # Use already uploaded PDF
            logger.info("📄 Generating rules from existing PDF", 
                       filename=pdf_filename,
                       chunk_size=chunk_size,
                       rule_types=rule_types_list)
            
            # Check if PDF exists
            logger.info("🔍 Checking if PDF exists...")
            pdf_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
            if not os.path.exists(pdf_path):
                logger.error("❌ PDF file not found", filename=pdf_filename, path=pdf_path)
                raise HTTPException(
                    status_code=404, 
                    detail=f"PDF '{pdf_filename}' not found"
                )
            logger.info("✅ PDF file found")
            
            # Generate rules and maintenance data
            logger.info("🚀 Starting rules generation from existing PDF...")
            result = await rules_generator.generate_rules_from_pdf(
                pdf_filename=pdf_filename,
                chunk_size=chunk_size,
                rule_types=rule_types_list
            )
        
        logger.info("🎉 Rules generation completed successfully", 
                   filename=result["pdf_filename"],
                   rules_count=len(result["iot_rules"]),
                   maintenance_count=len(result["maintenance_data"]),
                   safety_count=len(result["safety_precautions"]),
                   processing_time=f"{result['processing_time']:.2f}s")
        
        return RulesResponse(
            pdf_filename=result["pdf_filename"],
            total_pages=result["total_pages"],
            processed_chunks=result["processed_chunks"],
            iot_rules=result["iot_rules"],
            maintenance_data=result["maintenance_data"],
            safety_precautions=result["safety_precautions"],
            processing_time=result["processing_time"],
            summary=result["summary"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Rules generation failed", 
                    filename=file.filename if file else pdf_filename,
                    error=str(e))
        import traceback
        logger.error("❌ Full error traceback:", traceback=traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate rules: {str(e)}"
        )


@app.delete("/pdfs/{pdf_filename}", tags=["PDF Management"])
async def delete_pdf(pdf_filename: str):
    """
    Delete a PDF and all associated data
    
    - **pdf_filename**: Name of the PDF to delete
    
    Removes PDF file, ChromaDB data, and Minieu output
    """
    try:
        logger.info("Deleting PDF", filename=pdf_filename)
        
        pdf_name = os.path.splitext(pdf_filename)[0]
        success = False
        
        # Check if PDF exists in Minieu output
        minieu_output_dir = os.path.join(settings.MINIEU_OUTPUT_DIR, pdf_name)
        if os.path.exists(minieu_output_dir):
            success = True
            
            # Delete Minieu output directory
            import shutil
            shutil.rmtree(minieu_output_dir)
            logger.info("Deleted Minieu output directory", path=minieu_output_dir)
        
        # Delete ChromaDB data
        if chromadb_manager.delete_collection(pdf_name):
            logger.info("Deleted ChromaDB collection", pdf_name=pdf_name)
            success = True
        
        if success:
            # Delete uploaded file
            file_path = os.path.join(settings.UPLOAD_DIR, pdf_filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info("Deleted PDF file", path=file_path)
            
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