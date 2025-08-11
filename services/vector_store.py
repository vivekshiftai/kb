import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import asyncio
import os

from config.settings import get_settings
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Import vector store clients conditionally
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available - will use ChromaDB fallback")

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.error("ChromaDB not available - vector store will not work")


class BaseVectorStore:
    """Base class for vector store implementations"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    async def initialize(self):
        """Initialize the vector store"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check if vector store is accessible"""
        raise NotImplementedError
    
    async def store_document_chunks(self, pdf_filename: str, chunks: List[Dict[str, Any]], 
                                  file_hash: str) -> str:
        """Store document chunks"""
        raise NotImplementedError
    
    async def search_pdf(self, pdf_filename: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant content within a specific PDF"""
        raise NotImplementedError
    
    async def list_processed_pdfs(self) -> List[str]:
        """Get list of all processed PDF filenames"""
        raise NotImplementedError
    
    async def delete_pdf(self, pdf_filename: str) -> bool:
        """Delete all vectors for a specific PDF"""
        raise NotImplementedError
    
    async def get_pdf_stats(self, pdf_filename: str) -> Dict[str, Any]:
        """Get statistics for a specific PDF"""
        raise NotImplementedError


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, embedding_service: EmbeddingService):
        super().__init__(embedding_service)
        self.settings = get_settings()
        self.pc = None
        self.index = None
        
    async def initialize(self):
        """Initialize Pinecone client and index"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available")
            
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
            
            # Create index if it doesn't exist
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.settings.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.settings.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=self.settings.PINECONE_INDEX_NAME,
                    dimension=self.settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.settings.PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                await asyncio.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)
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
            logger.error(f"Pinecone health check failed: {e}")
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
            logger.error(f"Error storing document chunks in Pinecone: {e}")
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
            logger.error(f"Error searching PDF {pdf_filename} in Pinecone: {e}")
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
            
            logger.info(f"Found {len(pdf_filenames)} processed PDFs in Pinecone")
            return sorted(pdf_filenames)
            
        except Exception as e:
            logger.error(f"Error listing processed PDFs from Pinecone: {e}")
            return []

    async def delete_pdf(self, pdf_filename: str) -> bool:
        """Delete all vectors for a specific PDF"""
        try:
            namespace = f"pdf_{pdf_filename.replace('.pdf', '').replace(' ', '_')}"
            
            # Delete all vectors in the namespace
            self.index.delete(delete_all=True, namespace=namespace)
            
            logger.info(f"Deleted all vectors for {pdf_filename} from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting PDF {pdf_filename} from Pinecone: {e}")
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
                "namespace": namespace,
                "store_type": "pinecone"
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF stats from Pinecone: {e}")
            return {"pdf_filename": pdf_filename, "vector_count": 0, "store_type": "pinecone"}


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, embedding_service: EmbeddingService):
        super().__init__(embedding_service)
        self.settings = get_settings()
        self.client = None
        self.collection = None
        
    async def initialize(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB client not available")
            
        try:
            # Ensure persist directory exists
            os.makedirs(self.settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.settings.CHROMA_COLLECTION_NAME
                )
                logger.info(f"Using existing ChromaDB collection: {self.settings.CHROMA_COLLECTION_NAME}")
            except:
                self.collection = self.client.create_collection(
                    name=self.settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "PDF RAG collection"}
                )
                logger.info(f"Created new ChromaDB collection: {self.settings.CHROMA_COLLECTION_NAME}")
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            if self.collection is None:
                return False
            
            # Try to get collection count
            count = self.collection.count()
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

    async def store_document_chunks(self, pdf_filename: str, chunks: List[Dict[str, Any]], 
                                  file_hash: str) -> str:
        """Store document chunks in ChromaDB"""
        try:
            document_id = str(uuid.uuid4())
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
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
                    "images": json.dumps(chunk.get("images", [])),
                    "upload_date": datetime.now().isoformat(),
                    "chunk_type": "text"
                }
                
                ids.append(chunk_id)
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)
                documents.append(chunk["text"][:1000])  # Limit text length
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Stored {len(ids)} chunks for {pdf_filename} in ChromaDB")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing document chunks in ChromaDB: {e}")
            raise

    async def search_pdf(self, pdf_filename: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant content within a specific PDF"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.encode_text(query)
            
            # Search in ChromaDB with PDF filename filter
            search_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"pdf_filename": pdf_filename},
                include=["metadatas", "documents", "distances"]
            )
            
            # Process results
            results = {
                "matches": [],
                "total_matches": len(search_results["ids"][0]) if search_results["ids"] else 0
            }
            
            if search_results["ids"] and search_results["ids"][0]:
                for i, doc_id in enumerate(search_results["ids"][0]):
                    metadata = search_results["metadatas"][0][i]
                    document = search_results["documents"][0][i]
                    distance = search_results["distances"][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    score = 1.0 / (1.0 + distance)
                    
                    # Parse images from metadata
                    images = []
                    try:
                        images = json.loads(metadata.get("images", "[]"))
                    except:
                        pass
                    
                    results["matches"].append({
                        "id": doc_id,
                        "score": score,
                        "heading": metadata.get("heading", ""),
                        "text": document,
                        "images": images,
                        "chunk_index": metadata.get("chunk_index", 0)
                    })
            
            logger.info(f"Found {len(results['matches'])} matches for query in {pdf_filename}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching PDF {pdf_filename} in ChromaDB: {e}")
            raise

    async def list_processed_pdfs(self) -> List[str]:
        """Get list of all processed PDF filenames"""
        try:
            # Get all unique PDF filenames from the collection
            all_results = self.collection.get(
                include=["metadatas"],
                limit=10000  # Adjust based on expected size
            )
            
            pdf_filenames = set()
            for metadata in all_results["metadatas"]:
                if metadata and "pdf_filename" in metadata:
                    pdf_filenames.add(metadata["pdf_filename"])
            
            logger.info(f"Found {len(pdf_filenames)} processed PDFs in ChromaDB")
            return sorted(list(pdf_filenames))
            
        except Exception as e:
            logger.error(f"Error listing processed PDFs from ChromaDB: {e}")
            return []

    async def delete_pdf(self, pdf_filename: str) -> bool:
        """Delete all vectors for a specific PDF"""
        try:
            # Delete all documents for this PDF
            self.collection.delete(
                where={"pdf_filename": pdf_filename}
            )
            
            logger.info(f"Deleted all vectors for {pdf_filename} from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting PDF {pdf_filename} from ChromaDB: {e}")
            return False

    async def get_pdf_stats(self, pdf_filename: str) -> Dict[str, Any]:
        """Get statistics for a specific PDF"""
        try:
            # Count documents for this PDF
            count = self.collection.count(
                where={"pdf_filename": pdf_filename}
            )
            
            return {
                "pdf_filename": pdf_filename,
                "vector_count": count,
                "store_type": "chromadb"
            }
            
        except Exception as e:
            logger.error(f"Error getting PDF stats from ChromaDB: {e}")
            return {"pdf_filename": pdf_filename, "vector_count": 0, "store_type": "chromadb"}


class VectorStore:
    """Main vector store service with automatic fallback logic"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self._store = None
        self._store_type = None
        
    async def _initialize_store(self):
        """Initialize the appropriate vector store based on configuration"""
        if self._store is not None:
            return
            
        # Try Pinecone first if configured
        if self.settings.use_pinecone and PINECONE_AVAILABLE:
            try:
                logger.info("Attempting to initialize Pinecone vector store")
                self._store = PineconeVectorStore(self.embedding_service)
                await self._store.initialize()
                self._store_type = "pinecone"
                logger.info("Pinecone vector store initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Pinecone: {e}. Falling back to ChromaDB.")
                # Reset store to try ChromaDB
                self._store = None
        
        # Fallback to ChromaDB
        if CHROMADB_AVAILABLE:
            logger.info("Initializing ChromaDB vector store (fallback)")
            self._store = ChromaDBVectorStore(self.embedding_service)
            await self._store.initialize()
            self._store_type = "chromadb"
            logger.info("ChromaDB vector store initialized successfully")
        else:
            raise RuntimeError("No vector store available. Please install either Pinecone or ChromaDB.")
    
    async def initialize(self):
        """Initialize the vector store"""
        await self._initialize_store()
    
    async def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            await self._initialize_store()
            return await self._store.health_check()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False

    async def store_document_chunks(self, pdf_filename: str, chunks: List[Dict[str, Any]], 
                                  file_hash: str) -> str:
        """Store document chunks"""
        await self._initialize_store()
        return await self._store.store_document_chunks(pdf_filename, chunks, file_hash)
    
    async def search_pdf(self, pdf_filename: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for relevant content within a specific PDF"""
        await self._initialize_store()
        return await self._store.search_pdf(pdf_filename, query, top_k)
    
    async def list_processed_pdfs(self) -> List[str]:
        """Get list of all processed PDF filenames"""
        await self._initialize_store()
        return await self._store.list_processed_pdfs()
    
    async def delete_pdf(self, pdf_filename: str) -> bool:
        """Delete all vectors for a specific PDF"""
        await self._initialize_store()
        return await self._store.delete_pdf(pdf_filename)
    
    async def get_pdf_stats(self, pdf_filename: str) -> Dict[str, Any]:
        """Get statistics for a specific PDF"""
        await self._initialize_store()
        return await self._store.get_pdf_stats(pdf_filename)
    
    @property
    def store_type(self) -> str:
        """Get the current vector store type"""
        return self._store_type or self.settings.vector_store_type