#!/usr/bin/env python3
"""
Script to download required AI models for PDF processing
"""

import os
import sys
import time
import logging
from requests.exceptions import ConnectionError, HTTPError
from huggingface_hub import snapshot_download

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_with_retry(repo_id: str, local_dir: str, repo_type: str = "model",
                        max_workers: int = 1, resume_download: bool = True,
                        retries: int = 5, backoff_factor: float = 1.0) -> str:
    """Download model with retry mechanism"""
    
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries}: downloading {repo_id}...")
            
            # Ensure directory exists
            os.makedirs(local_dir, exist_ok=True)
            
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type=repo_type,
                max_workers=max_workers,
                resume_download=resume_download
            )
            
            logger.info(f"Download succeeded: {path}")
            return path
            
        except (ConnectionError, HTTPError) as e:
            wait_time = backoff_factor * (2 ** (attempt - 1))
            logger.error(f"Error on attempt {attempt}: {e}")
            
            if attempt < retries:
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to download {repo_id} after {retries} attempts")
        
        except Exception as e:
            logger.error(f"Unexpected error downloading {repo_id}: {e}")
            raise

def download_pdf_extract_kit():
    """Download PDF-Extract-Kit model"""
    settings = get_settings()
    
    logger.info("Downloading PDF-Extract-Kit-1.0 model...")
    
    try:
        path = download_with_retry(
            repo_id="opendatalab/pdf-extract-kit-1.0",
            local_dir=settings.MODELS_DIR,
            repo_type="model"
        )
        logger.info(f"PDF-Extract-Kit downloaded to: {path}")
        return path
        
    except Exception as e:
        logger.error(f"Failed to download PDF-Extract-Kit: {e}")
        raise

def download_sentence_transformers():
    """Download sentence transformer models"""
    settings = get_settings()
    
    models = [
        settings.TEXT_EMBEDDING_MODEL,
        settings.IMAGE_EMBEDDING_MODEL
    ]
    
    for model in models:
        try:
            logger.info(f"Downloading sentence transformer model: {model}")
            model_dir = os.path.join(settings.MODELS_DIR, "sentence_transformers", model.replace("/", "_"))
            
            path = download_with_retry(
                repo_id=model,
                local_dir=model_dir,
                repo_type="model"
            )
            logger.info(f"Model {model} downloaded to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to download model {model}: {e}")
            # Continue with other models even if one fails
            continue

def verify_downloads():
    """Verify that all required models are downloaded"""
    settings = get_settings()
    
    required_paths = [
        settings.MODELS_DIR,
        os.path.join(settings.MODELS_DIR, "sentence_transformers")
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        logger.warning(f"Missing model directories: {missing_paths}")
        return False
    
    logger.info("All required model directories exist")
    return True

def main():
    """Main function to download all required models"""
    try:
        logger.info("Starting model download process...")
        
        # Create models directory
        settings = get_settings()
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        
        # Download PDF extraction models
        try:
            download_pdf_extract_kit()
        except Exception as e:
            logger.error(f"PDF-Extract-Kit download failed: {e}")
            logger.info("Continuing with other models...")
        
        # Download sentence transformer models
        try:
            download_sentence_transformers()
        except Exception as e:
            logger.error(f"Sentence transformer download failed: {e}")
        
        # Verify downloads
        if verify_downloads():
            logger.info("Model download process completed successfully!")
        else:
            logger.warning("Some models may not have been downloaded correctly")
            
    except Exception as e:
        logger.error(f"Model download process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()