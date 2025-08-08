import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import logging
from typing import List, Union
import torch

from config.settings import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text and image embeddings"""
    
    def __init__(self):
        self.settings = get_settings()
        self.text_model = None
        self.image_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        try:
            # Initialize text embedding model
            logger.info(f"Loading text embedding model: {self.settings.TEXT_EMBEDDING_MODEL}")
            self.text_model = SentenceTransformer(self.settings.TEXT_EMBEDDING_MODEL)
            
            # Initialize image embedding model
            logger.info(f"Loading image embedding model: {self.settings.IMAGE_EMBEDDING_MODEL}")
            self.image_model = SentenceTransformer(self.settings.IMAGE_EMBEDDING_MODEL)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() and self.settings.DEVICE_MODE == "cuda" else "cpu"
            self.text_model = self.text_model.to(device)
            self.image_model = self.image_model.to(device)
            
            logger.info(f"Embedding models loaded successfully on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            raise

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        try:
            if not text or not text.strip():
                # Return zero embedding for empty text
                return np.zeros(self.text_model.get_sentence_embedding_dimension())
            
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.text_model.get_sentence_embedding_dimension())

    async def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embedding vectors"""
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                # Return zero embeddings
                dim = self.text_model.get_sentence_embedding_dimension()
                return np.zeros((len(texts), dim))
            
            embeddings = self.text_model.encode(valid_texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            # Return zero embeddings as fallback
            dim = self.text_model.get_sentence_embedding_dimension()
            return np.zeros((len(texts), dim))

    async def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image to embedding vector"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Generate embedding
            embedding = self.image_model.encode(image, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.image_model.get_sentence_embedding_dimension())

    async def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode multiple images to embedding vectors"""
        try:
            images = []
            valid_paths = []
            
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
            
            if not images:
                # Return zero embeddings
                dim = self.image_model.get_sentence_embedding_dimension()
                return np.zeros((len(image_paths), dim))
            
            embeddings = self.image_model.encode(images, convert_to_numpy=True)
            
            # If some images failed to load, pad with zeros
            if len(embeddings) < len(image_paths):
                dim = embeddings.shape[1]
                full_embeddings = np.zeros((len(image_paths), dim))
                full_embeddings[:len(embeddings)] = embeddings
                return full_embeddings
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            # Return zero embeddings as fallback
            dim = self.image_model.get_sentence_embedding_dimension()
            return np.zeros((len(image_paths), dim))

    def get_text_embedding_dimension(self) -> int:
        """Get text embedding dimension"""
        return self.text_model.get_sentence_embedding_dimension()

    def get_image_embedding_dimension(self) -> int:
        """Get image embedding dimension"""
        return self.image_model.get_sentence_embedding_dimension()

    async def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0