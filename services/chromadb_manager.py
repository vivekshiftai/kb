import os
import logging
import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Centralized ChromaDB manager for handling all PDF collections"""
    
    def __init__(self):
        self.settings = get_settings()
        self.chromadb_path = "./chroma_db"
        self.client = chromadb.PersistentClient(path=self.chromadb_path)
        self.text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def get_collection_name(self, pdf_name: str) -> str:
        """Get collection name for a PDF"""
        return f"pdf_{pdf_name}"
    
    def create_collection(self, pdf_name: str) -> chromadb.Collection:
        """Create a new collection for a PDF"""
        collection_name = self.get_collection_name(pdf_name)
        
        try:
            collection = self.client.get_or_create_collection(collection_name)
            logger.info(f"Created/retrieved collection: {collection_name}")
            return collection
        except Exception as e:
            logger.warning(f"Primary collection creation failed, trying v1.0.x format: {e}")
            try:
                # Fallback for ChromaDB v1.0.x
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created collection with v1.0.x format: {collection_name}")
                return collection
            except Exception as e2:
                logger.error(f"ChromaDB collection creation failed: {e2}")
                raise Exception(f"ChromaDB collection creation failed: {e2}")
    
    def get_collection(self, pdf_name: str) -> Optional[chromadb.Collection]:
        """Get existing collection for a PDF"""
        collection_name = self.get_collection_name(pdf_name)
        
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except Exception as e:
            logger.warning(f"Collection not found: {collection_name}")
            return None
    
    def collection_exists(self, pdf_name: str) -> bool:
        """Check if collection exists for a PDF"""
        collection = self.get_collection(pdf_name)
        return collection is not None and collection.count() > 0
    
    def store_chunks(self, pdf_name: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store chunks in the PDF's collection"""
        try:
            collection = self.create_collection(pdf_name)
            
            for i, chunk in enumerate(chunks):
                # Create combined text for embedding
                combined_text = f"{chunk['heading']}\n{chunk['text']}"
                embedding = self.text_embedder.encode(combined_text).tolist()
                
                # Prepare metadata
                metadata = {
                    "heading": chunk["heading"],
                    "images": ";".join(chunk["images"]),
                    "tables_count": len(chunk.get("tables", [])),
                    "page_number": chunk.get("page_number", 1),
                    "source_file": chunk.get("source_file", f"{pdf_name}.pdf"),
                    "pdf_name": pdf_name,
                    "chunk_type": "text"
                }
                
                # Add text chunk
                collection.add(
                    ids=[f"{pdf_name}_text_{i}"],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[chunk["text"]]
                )
                
                # Add table chunks if they exist
                for j, table in enumerate(chunk.get("tables", [])):
                    table_text = f"Table: {table.get('title', 'Untitled')}\n{table.get('content', '')}"
                    table_embedding = self.text_embedder.encode(table_text).tolist()
                    
                    table_metadata = {
                        "heading": chunk["heading"],
                        "table_title": table.get('title', 'Untitled'),
                        "page_number": chunk.get("page_number", 1),
                        "source_file": chunk.get("source_file", f"{pdf_name}.pdf"),
                        "pdf_name": pdf_name,
                        "chunk_type": "table"
                    }
                    
                    collection.add(
                        ids=[f"{pdf_name}_table_{i}_{j}"],
                        embeddings=[table_embedding],
                        metadatas=[table_metadata],
                        documents=[table_text]
                    )
            
            logger.info(f"Stored {len(chunks)} chunks in collection for {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks for {pdf_name}: {e}")
            return False
    
    def search(self, pdf_name: str, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search in a specific PDF's collection"""
        try:
            collection = self.get_collection(pdf_name)
            if not collection:
                logger.warning(f"Collection not found for {pdf_name}")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            results = collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["metadatas", "documents"]
            )
            
            logger.info(f"Found {len(results['documents'][0])} results for query in {pdf_name}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for {pdf_name}: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def list_collections(self) -> List[str]:
        """List all PDF collections"""
        try:
            collections = self.client.list_collections()
            pdf_collections = [col.name for col in collections if col.name.startswith("pdf_")]
            logger.info(f"Found {len(pdf_collections)} PDF collections")
            return pdf_collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def delete_collection(self, pdf_name: str) -> bool:
        """Delete a PDF's collection"""
        try:
            collection_name = self.get_collection_name(pdf_name)
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection for {pdf_name}: {e}")
            return False
    
    def get_collection_stats(self, pdf_name: str) -> Dict[str, Any]:
        """Get statistics for a PDF's collection"""
        try:
            collection = self.get_collection(pdf_name)
            if not collection:
                return {"exists": False, "count": 0}
            
            count = collection.count()
            return {
                "exists": True,
                "count": count,
                "name": collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {pdf_name}: {e}")
            return {"exists": False, "count": 0, "error": str(e)}
