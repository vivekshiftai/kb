from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for querying documents"""
    question: str = Field(..., description="Question to ask about the documents")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search in")

class Source(BaseModel):
    """Source information for query results"""
    heading: str
    document: str
    images: List[str] = []

class QueryResponse(BaseModel):
    """Response model for document queries"""
    question: str
    answer: str
    sources: List[Source] = []
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)

class DocumentInfo(BaseModel):
    """Document information model"""
    id: str
    filename: str
    upload_date: datetime
    chunk_count: int
    status: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    vector_store_status: str
    openai_status: str

class ProcessingStatus(BaseModel):
    """Processing status model"""
    document_id: str
    filename: str
    status: str  # "processing", "completed", "failed"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

class ChunkData(BaseModel):
    """Chunk data model"""
    id: str
    text: str
    heading: str
    images: List[str] = []
    tables: List[str] = []
    metadata: Dict[str, Any] = {}

class DocumentMetadata(BaseModel):
    """Document metadata model"""
    filename: str
    file_hash: str
    file_size: int
    page_count: int
    upload_date: datetime
    processing_date: Optional[datetime] = None
    chunk_count: int = 0
    image_count: int = 0
    table_count: int = 0