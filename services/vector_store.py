import pinecone
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from config.settings import get_settings
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStore:
    """Pinecone vector store service"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.index = None
        
    async def initialize(self):
        """Initialize Pinecone client and index"""
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.settings.PINECONE_API_KEY,
                environment=self.settings.PINECONE_ENVIRONMENT
            )
            
            # Create index if it doesn't exist
            if self.settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.settings.PINECONE_INDEX_NAME}")
                pinecone.create_index(
                    name=self.settings.PINECONE_INDEX_NAME,
                    dimension=self.settings.EMBEDDING_DIMENSION,
                    metric="cosine"
                )
            
            # Connect to index
            self.index = pinecone.Index(self.settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            if self.index is None:
                return False
            
            # Try to get index stats
            stats = self.index.describe_index_stats()
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False

    async def store_document_chunks(self, pdf_filename: str, chunks: List[Dict[str, Any]], 
                                  file_hash: str) -> str:
        """Store document chunks in Pinecone with PDF-specific namespace"""
        try:
            document_id = str(uuid.uuid4())
            namespace = f"pdf_{pdf_filename.replace('.pdf', '').replace(' ', '_')}"
            
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                if not chunk.get("text", "").strip():
                    continue
                    
                # Generate embedding for the chunk
                combined_text = f"{chunk.get('heading', '')}\n{chunk['text']}"
                embedding = await self.embedding_service.encode_text(combined_text)
                
                # Create unique ID for this chunk
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Prepare metadata
                metadata = {
                    "document_id": document_id,
                    "pdf_filename": pdf_filename,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "heading": chunk.get("heading", ""),
                    "text": chunk["text"][:1000],  # Limit text in metadata
                    "images": json.dumps(chunk.get("images", [])),
                    "upload_date": datetime.now().isoformat(),
                    "chunk_type": "text"
                }
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                logger.info(f"Upserted batch {i//batch_size + 1} for {pdf_filename}")
            
            logger.info(f"Stored {len(vectors_to_upsert)} chunks for {pdf_filename} in namespace {namespace}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {e}")
            raise

    async def search_pdf(self, pdf_filename: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant content within a specific PDF"""
        try:
            namespace = f"pdf_{pdf_filename.replace('.pdf', '').replace(' ', '_')}"
            
            # Generate query embedding
            query_embedding = await self.embedding_service.encode_text(query)
            
            # Search in the PDF-specific namespace
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            # Process results
            results = {
                "matches": [],
                "total_matches": len(search_results.matches)
            }
            
            for match in search_results.matches:
                metadata = match.metadata
                
                # Parse images from metadata
                images = []
                try:
                    images = json.loads(metadata.get("images", "[]"))
                except:
                    pass
                
                results["matches"].append({
                    "id": match.id,
                    "score": match.score,
                    "heading": metadata.get("heading", ""),
                    "text": metadata.get("text", ""),
                    "images": images,
                    "chunk_index": metadata.get("chunk_index", 0)
                })
            
            logger.info(f"Found {len(results['matches'])} matches for query in {pdf_filename}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PDF {pdf_filename}: {e}")
            raise

    async def list_processed_pdfs(self) -> List[str]:
        """Get list of all processed PDF filenames"""
        try:
            # Get index stats to see all namespaces
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            
            # Extract PDF filenames from namespaces
            pdf_filenames = []
            for namespace in namespaces:
                if namespace.startswith("pdf_"):
                    # Convert namespace back to filename
                    filename = namespace.replace("pdf_", "").replace("_", " ") + ".pdf"
                    pdf_filenames.append(filename)
            
            logger.info(f"Found {len(pdf_filenames)} processed PDFs")
            return sorted(pdf_filenames)
            
        except Exception as e:
            logger.error(f"Error listing processed PDFs: {e}")
            return []

    async def delete_pdf(self, pdf_filename: str) -> bool:
        """Delete all vectors for a specific PDF"""
        try:
            namespace = f"pdf_{pdf_filename.replace('.pdf', '').replace(' ', '_')}"
            
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=namespace)
            
            logger.info(f"Deleted all vectors for {pdf_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting PDF {pdf_filename}: {e}")
            return False

    async def get_pdf_stats(self, pdf_filename: str) -> Dict[str, Any]:
        """Get statistics for a specific PDF"""
        try:
            namespace = f"pdf_{pdf_filename.replace('.pdf', '').replace(' ', '_')}"
            
            # Get namespace stats
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(namespace, {})
            
            return {
                "pdf_filename": pdf_filename,
                "vector_count": namespace_stats.get("vector_count", 0),
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF stats: {e}")
            return {"pdf_filename": pdf_filename, "vector_count": 0}