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

class IoTDeviceRule(BaseModel):
    """IoT device rule model"""
    device_name: str
    rule_type: str  # "monitoring", "maintenance", "alert", "control"
    condition: str
    action: str
    priority: str = "medium"  # "low", "medium", "high", "critical"
    frequency: Optional[str] = None  # "hourly", "daily", "weekly", "monthly"
    description: str

class MaintenanceData(BaseModel):
    """Maintenance data model"""
    component_name: str
    maintenance_type: str  # "preventive", "corrective", "predictive"
    frequency: str
    last_maintenance: Optional[str] = None
    next_maintenance: Optional[str] = None
    description: str

class SafetyPrecaution(BaseModel):
    """Safety precaution model"""
    category: str  # "electrical safety", "mechanical safety", "chemical safety", "fire safety", "personal protection"
    precaution: str
    equipment: Optional[str] = None  # Required safety equipment or PPE
    procedure: Optional[str] = None  # Safety procedure or protocol
    warning_level: str = "medium"  # "low", "medium", "high", "critical"
    description: str

class RulesRequest(BaseModel):
    """Request model for generating rules from PDF"""
    pdf_filename: str = Field(..., description="Name of the PDF file to analyze")
    chunk_size: Optional[int] = Field(10, description="Number of pages to process in each chunk", ge=1, le=50)
    rule_types: Optional[List[str]] = Field(["monitoring", "maintenance", "alert"], description="Types of rules to generate")

class RulesResponse(BaseModel):
    """Response model for rules generation"""
    pdf_filename: str
    total_pages: int
    processed_chunks: int
    iot_rules: List[IoTDeviceRule]
    maintenance_data: List[MaintenanceData]
    safety_precautions: List[SafetyPrecaution]
    processing_time: float
    summary: str