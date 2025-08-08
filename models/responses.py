from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class APIResponse(BaseModel):
    """Base API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    """PDF upload response model"""
    success: bool
    message: str
    document_id: Optional[str] = None
    filename: str
    status: str

class SearchResult(BaseModel):
    """Search result model"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class EmbeddingResult(BaseModel):
    """Embedding generation result"""
    success: bool
    embeddings: List[float]
    model_used: str
    dimension: int