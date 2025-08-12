import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List
import torch
import asyncio

from config.settings import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.settings.TEXT_EMBEDDING_MODEL}")
            self.model = SentenceTransformer(self.settings.TEXT_EMBEDDING_MODEL)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            logger.info(f"Embedding model loaded successfully on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        try:
            if not text or not text.strip():
                # Return zero embedding for empty text
                return np.zeros(self.settings.EMBEDDING_DIMENSION)
            
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(text, convert_to_numpy=True)
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.settings.EMBEDDING_DIMENSION)

    async def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embedding vectors"""
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                # Return zero embeddings
                return np.zeros((len(texts), self.settings.EMBEDDING_DIMENSION))
            
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(valid_texts, convert_to_numpy=True)
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.settings.EMBEDDING_DIMENSION))

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.settings.EMBEDDING_DIMENSION