import chromadb
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from config.settings import get_settings
from services.embeddings import EmbeddingService
from models.schemas import DocumentMetadata, ChunkData

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB vector store service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.text_collection = None
        self.image_collection = None
        self.embedding_service = EmbeddingService()
        
    async def initialize(self):
        """Initialize ChromaDB client and collections"""
        try:
            # Create ChromaDB directory if it doesn't exist
            os.makedirs(self.settings.CHROMADB_DIR, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(path=self.settings.CHROMADB_DIR)
            
            # Create or get collections
            self.text_collection = self.client.get_or_create_collection(
                name=self.settings.COLLECTION_NAME,
                metadata={"description": "PDF document text chunks"}
            )
            
            self.image_collection = self.client.get_or_create_collection(
                name=self.settings.IMAGE_COLLECTION_NAME,
                metadata={"description": "PDF document images"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def close(self):
        """Close ChromaDB connections"""
        # ChromaDB doesn't require explicit closing
        logger.info("ChromaDB connections closed")

    async def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            if self.client is None:
                return False
            
            # Try to get collection info
            collections = self.client.list_collections()
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False

    async def store_document(self, file_path: str, filename: str, 
                           file_hash: str, output_dir: str) -> str:
        """Store processed document in vector database"""
        try:
            document_id = str(uuid.uuid4())
            
            # Import here to avoid circular import
            from services.pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor()
            
            # Get processing results
            results = await pdf_processor.get_processing_results(output_dir, file_path)
            chunks = results["chunks"]
            image_files = results["images"]
            
            # Extract PDF metadata
            metadata = await pdf_processor.extract_pdf_metadata(file_path)
            
            # Store text chunks
            chunk_count = 0
            for i, chunk in enumerate(chunks):
                if chunk["text"].strip():  # Only store non-empty chunks
                    chunk_id = f"{document_id}_chunk_{i}"
                    
                    # Generate embedding
                    combined_text = f"{chunk['heading']}\n{chunk['text']}"
                    embedding = await self.embedding_service.encode_text(combined_text)
                    
                    # Prepare metadata
                    chunk_metadata = {
                        "document_id": document_id,
                        "filename": filename,
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "heading": chunk["heading"],
                        "images": ";".join(chunk["images"]) if chunk["images"] else "",
                        "tables": json.dumps(chunk["tables"]) if chunk["tables"] else "",
                        "upload_date": datetime.now().isoformat(),
                        "page_count": metadata.get("page_count", 0),
                        "file_size": metadata.get("file_size", 0)
                    }
                    
                    # Add to collection
                    self.text_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding.tolist()],
                        documents=[chunk["text"]],
                        metadatas=[chunk_metadata]
                    )
                    
                    chunk_count += 1
            
            # Store images
            image_count = 0
            for i, img_path in enumerate(image_files):
                try:
                    image_id = f"{document_id}_image_{i}"
                    
                    # Generate image embedding
                    embedding = await self.embedding_service.encode_image(img_path)
                    
                    # Prepare metadata
                    img_metadata = {
                        "document_id": document_id,
                        "filename": filename,
                        "image_path": img_path,
                        "image_index": i,
                        "upload_date": datetime.now().isoformat()
                    }
                    
                    # Add to image collection
                    self.image_collection.add(
                        ids=[image_id],
                        embeddings=[embedding.tolist()],
                        documents=[f"[IMAGE] {os.path.basename(img_path)}"],
                        metadatas=[img_metadata]
                    )
                    
                    image_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {img_path}: {e}")
            
            logger.info(f"Stored document {document_id}: {chunk_count} chunks, {image_count} images")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document in vector store: {e}")
            raise

    async def search(self, query: str, n_results: int = 5, 
                    document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search for relevant content"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.encode_text(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            # Search in text collection
            results = self.text_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    async def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get document by file hash"""
        try:
            results = self.text_collection.query(
                query_embeddings=[[0] * 384],  # Dummy embedding
                n_results=1,
                where={"file_hash": file_hash},
                include=["metadatas"]
            )
            
            if results["metadatas"] and results["metadatas"][0]:
                return results["metadatas"][0][0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting document by hash: {e}")
            return None

    async def list_documents(self) -> List[Dict[str, Any]]:
        """Get list of all documents"""
        try:
            # Get all unique documents
            all_results = self.text_collection.get(include=["metadatas"])
            
            # Group by document_id
            documents = {}
            for metadata in all_results["metadatas"]:
                doc_id = metadata["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "filename": metadata["filename"],
                        "upload_date": metadata["upload_date"],
                        "chunk_count": 0,
                        "status": "completed"
                    }
                documents[doc_id]["chunk_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        try:
            # Get all chunk IDs for this document
            results = self.text_collection.get(
                where={"document_id": document_id},
                include=["ids"]
            )
            
            if results["ids"]:
                # Delete text chunks
                self.text_collection.delete(
                    where={"document_id": document_id}
                )
                
                # Delete images
                self.image_collection.delete(
                    where={"document_id": document_id}
                )
                
                logger.info(f"Deleted document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    async def get_document_chunks(self, document_id: str) -> List[ChunkData]:
        """Get all chunks for a document"""
        try:
            results = self.text_collection.get(
                where={"document_id": document_id},
                include=["ids", "documents", "metadatas"]
            )
            
            chunks = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                chunks.append(ChunkData(
                    id=results["ids"][i],
                    text=results["documents"][i],
                    heading=metadata.get("heading", ""),
                    images=metadata.get("images", "").split(";") if metadata.get("images") else [],
                    tables=json.loads(metadata.get("tables", "[]")) if metadata.get("tables") else [],
                    metadata=metadata
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []