"""
Application settings and configuration management
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 1500
    OPENAI_TEMPERATURE: float = 0.7
    
    # File Processing Configuration
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB default
    ALLOWED_EXTENSIONS: list = [".pdf"]
    
    # Pinecone Vector Database Configuration (Optional)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"  # AWS region format
    PINECONE_INDEX_NAME: str = "pdf-rag-index"
    
    # ChromaDB Vector Database Configuration (Fallback)
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "pdf_rag_collection"
    
    # Embedding Configuration
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Text Processing Configuration
    CHUNK_MAX_LENGTH: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SEARCH_RESULTS: int = 10
    
    # Background Processing Configuration
    CLEANUP_INTERVAL_DAYS: int = 7
    MAX_CONCURRENT_PROCESSING: int = 3
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # API Configuration
    API_TITLE: str = "RAG PDF Processing API"
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Backend API for PDF processing and intelligent querying"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

    @property
    def use_pinecone(self) -> bool:
        """Check if Pinecone should be used based on credentials availability"""
        return bool(self.PINECONE_API_KEY)

    @property
    def vector_store_type(self) -> str:
        """Get the vector store type to use"""
        return "pinecone" if self.use_pinecone else "chromadb"


# Global settings singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()