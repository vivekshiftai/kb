"""
Application settings and configuration management
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import logging

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_ENV: str = Field(default="development", env="APP_ENV")
    DEBUG: bool = Field(default=True, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKER_PROCESSES")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    
    # File Storage Settings
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")
    OUTPUT_DIR: str = Field(default="./output", env="OUTPUT_DIR")
    MINIEU_OUTPUT_DIR: str = Field(default="./minieu_output", env="MINIEU_OUTPUT_DIR")
    MAX_FILE_SIZE: int = Field(default=52428800, env="MAX_FILE_SIZE")  # 50MB
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".pdf"], env="ALLOWED_EXTENSIONS")
    
    # Chunking Settings
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    MAX_CHUNKS_PER_PDF: int = Field(default=1000, env="MAX_CHUNKS_PER_PDF")
    CHUNK_MAX_LENGTH: int = Field(default=1000, env="CHUNK_MAX_LENGTH")
    
    # Search Settings
    MAX_SEARCH_RESULTS: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    
    # Vector Database Settings
    VECTOR_STORE_TYPE: str = Field(default="chromadb", env="VECTOR_STORE_TYPE")
    USE_PINECONE: bool = Field(default=False, env="USE_PINECONE")
    
    # Pinecone Settings
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field(default="kb-index", env="PINECONE_INDEX_NAME")
    
    # ChromaDB Settings
    CHROMADB_HOST: str = Field(default="localhost", env="CHROMADB_HOST")
    CHROMADB_PORT: int = Field(default=8000, env="CHROMADB_PORT")
    CHROMADB_PERSIST_DIRECTORY: str = Field(default="./chromadb", env="CHROMADB_PERSIST_DIRECTORY")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    OPENAI_TIMEOUT: int = Field(default=60, env="OPENAI_TIMEOUT")
    
    # Security Settings
    SECRET_KEY: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Performance Settings
    EMBEDDING_BATCH_SIZE: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    MAX_CONCURRENT_UPLOADS: int = Field(default=5, env="MAX_CONCURRENT_UPLOADS")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Monitoring Settings
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Backup Settings
    BACKUP_ENABLED: bool = Field(default=True, env="BACKUP_ENABLED")
    BACKUP_RETENTION_DAYS: int = Field(default=7, env="BACKUP_RETENTION_DAYS")
    BACKUP_SCHEDULE: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE")  # Daily at 2 AM
    
    # Error Handling Settings
    MAX_RETRY_ATTEMPTS: int = Field(default=3, env="MAX_RETRY_ATTEMPTS")
    RETRY_DELAY: int = Field(default=1, env="RETRY_DELAY")
    CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, env="CIRCUIT_BREAKER_THRESHOLD")
    
    # Image Processing Settings
    MAX_IMAGE_SIZE: int = Field(default=10485760, env="MAX_IMAGE_SIZE")  # 10MB
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(default=[".png", ".jpg", ".jpeg", ".gif", ".bmp"], env="SUPPORTED_IMAGE_FORMATS")
    IMAGE_COMPRESSION_QUALITY: int = Field(default=85, env="IMAGE_COMPRESSION_QUALITY")
    
    # Logging Settings
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: str = Field(default="./app.log", env="LOG_FILE")
    LOG_MAX_SIZE: int = Field(default=10485760, env="LOG_MAX_SIZE")  # 10MB
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Rate Limiting Settings
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from environment variables
    }

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
        
        # Ensure directories exist
        os.makedirs(_settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(_settings.OUTPUT_DIR, exist_ok=True)
        os.makedirs(_settings.MINIEU_OUTPUT_DIR, exist_ok=True)
        os.makedirs(_settings.CHROMADB_PERSIST_DIRECTORY, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(_settings.OUTPUT_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(_settings.OUTPUT_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(_settings.OUTPUT_DIR, "temp"), exist_ok=True)
        
        # Set proper permissions for Ubuntu
        if os.name == 'posix':  # Unix-like systems
            try:
                os.chmod(_settings.UPLOAD_DIR, 0o755)
                os.chmod(_settings.OUTPUT_DIR, 0o755)
                os.chmod(_settings.MINIEU_OUTPUT_DIR, 0o755)
                os.chmod(_settings.CHROMADB_PERSIST_DIRECTORY, 0o755)
            except PermissionError:
                logging.warning("Could not set directory permissions - running without elevated privileges")
    
    return _settings

def validate_settings() -> bool:
    """Validate critical settings"""
    settings = get_settings()
    
    # Check required API keys
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your-openai-api-key":
        logging.error("OpenAI API key not configured")
        return False
    
    # Check Pinecone settings if enabled
    if settings.USE_PINECONE:
        if not settings.PINECONE_API_KEY:
            logging.error("Pinecone API key required when USE_PINECONE is True")
            return False
        if not settings.PINECONE_ENVIRONMENT:
            logging.error("Pinecone environment required when USE_PINECONE is True")
            return False
    
    # Check directory permissions
    for directory in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.MINIEU_OUTPUT_DIR]:
        if not os.access(directory, os.W_OK):
            logging.error(f"Directory {directory} is not writable")
            return False
    
    return True

def get_environment_info() -> dict:
    """Get environment information for debugging"""
    settings = get_settings()
    
    return {
        "app_env": settings.APP_ENV,
        "debug": settings.DEBUG,
        "log_level": settings.LOG_LEVEL,
        "vector_store_type": settings.VECTOR_STORE_TYPE,
        "use_pinecone": settings.USE_PINECONE,
        "upload_dir": settings.UPLOAD_DIR,
        "output_dir": settings.OUTPUT_DIR,
        "max_file_size": settings.MAX_FILE_SIZE,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "openai_model": settings.OPENAI_MODEL,
        "workers": settings.WORKERS,
        "host": settings.HOST,
        "port": settings.PORT,
        "cors_origins": settings.CORS_ORIGINS,
        "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
        "backup_enabled": settings.BACKUP_ENABLED,
        "metrics_enabled": settings.ENABLE_METRICS
    }
