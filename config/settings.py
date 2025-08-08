from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 1500
    OPENAI_TEMPERATURE: float = 0.7
    
    # File Processing
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list = [".pdf"]
    
    # ChromaDB Configuration
    CHROMADB_DIR: str = "./chromadb_storage"
    COLLECTION_NAME: str = "pdf_documents"
    IMAGE_COLLECTION_NAME: str = "document_images"
    
    # MinerU Configuration
    DEVICE_MODE: str = "cuda"  # or "cpu"
    VIRTUAL_VRAM_SIZE: int = 8
    MODELS_DIR: str = "./models"
    
    # Embedding Models
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL: str = "clip-ViT-B-32"
    
    # Processing Configuration
    CHUNK_MAX_LENGTH: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SEARCH_RESULTS: int = 10
    
    # Background Task Configuration
    CLEANUP_INTERVAL_DAYS: int = 7
    MAX_CONCURRENT_PROCESSING: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings