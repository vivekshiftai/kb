from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for querying a specific PDF"""
    pdf_filename: str = Field(..., description="Name of the PDF file to query")
    query: str = Field(..., description="Question to ask about the PDF")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return", ge=1, le=20)

class ImageInfo(BaseModel):
    """Image information model"""
    filename: str
    url: str
    page_number: int

class QueryResult(BaseModel):
    """Individual query result"""
    heading: str
    text: str
    score: float
    page_number: Optional[int] = None
    images: List[ImageInfo] = []

class QueryResponse(BaseModel):
    """Response model for PDF queries"""
    pdf_filename: str
    query: str
    answer: str
    results: List[QueryResult] = []
    total_matches: int
    processing_time: float

class PDFInfo(BaseModel):
    """PDF information model"""
    filename: str
    upload_date: Optional[datetime] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    chunk_count: Optional[int] = None

class PDFListResponse(BaseModel):
    """Response model for listing PDFs"""
    pdfs: List[PDFInfo]
    total_count: int

class UploadResponse(BaseModel):
    """PDF upload response model"""
    success: bool
    message: str
    pdf_filename: str
    document_id: Optional[str] = None
    processing_status: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    vector_store_status: str
    openai_status: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime