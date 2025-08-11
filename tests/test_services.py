import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from services.embeddings import EmbeddingService
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStore
from config.settings import get_settings

class TestEmbeddingService:
    """Test embedding service"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing"""
        with patch('sentence_transformers.SentenceTransformer'):
            service = EmbeddingService()
            service.model = Mock()
            service.model.encode.return_value = np.random.rand(384)
            return service
    
    @pytest.mark.asyncio
    async def test_encode_text(self, embedding_service):
        """Test text encoding"""
        text = "This is a test sentence."
        embedding = await embedding_service.encode_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        embedding_service.model.encode.assert_called_once_with(text, convert_to_numpy=True)
    
    @pytest.mark.asyncio
    async def test_encode_empty_text(self, embedding_service):
        """Test encoding empty text"""
        embedding = await embedding_service.encode_text("")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert np.all(embedding == 0)
    
    @pytest.mark.asyncio
    async def test_encode_multiple_texts(self, embedding_service):
        """Test encoding multiple texts"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embedding_service.model.encode.return_value = np.random.rand(3, 384)
        
        embeddings = await embedding_service.encode_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

class TestPDFProcessor:
    """Test PDF processor"""
    
    @pytest.fixture
    def pdf_processor(self):
        """Create PDF processor for testing"""
        return PDFProcessor()
    
    @pytest.fixture
    def mock_pdf_doc(self):
        """Mock PDF document"""
        mock_doc = Mock()
        mock_doc.page_count = 2
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "creationDate": "2023-01-01"
        }
        
        # Mock pages
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "This is page 1 content."
        mock_page1.get_text.return_value = {
            "blocks": [{
                "lines": [{
                    "spans": [{
                        "text": "Test Heading",
                        "size": 16,
                        "flags": 16,
                        "bbox": [0, 0, 100, 20]
                    }]
                }]
            }]
        }
        mock_page1.get_images.return_value = []
        
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "This is page 2 content."
        mock_page2.get_text.return_value = {"blocks": []}
        mock_page2.get_images.return_value = []
        
        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        return mock_doc
    
    @pytest.mark.asyncio
    async def test_process_pdf(self, pdf_processor, mock_pdf_doc):
        """Test PDF processing"""
        with patch('fitz.open', return_value=mock_pdf_doc), \
             patch('os.path.getsize', return_value=1024), \
             patch('os.makedirs'):
            
            result = await pdf_processor.process_pdf("/test/path.pdf", "test.pdf")
            
            assert result["filename"] == "test.pdf"
            assert "metadata" in result
            assert "chunks" in result
            assert result["metadata"]["page_count"] == 2
    
    def test_extract_metadata(self, pdf_processor, mock_pdf_doc):
        """Test metadata extraction"""
        with patch('os.path.getsize', return_value=1024):
            metadata = pdf_processor._extract_metadata(mock_pdf_doc, "/test/path.pdf")
            
            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["page_count"] == 2
            assert metadata["file_size"] == 1024

class TestVectorStore:
    """Test vector store"""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store for testing"""
        with patch('pinecone.Pinecone') as mock_pc_class:
            mock_pc = Mock()
            mock_pc_class.return_value = mock_pc
            mock_pc.list_indexes.return_value = []
            
            store = VectorStore()
            store.pc = mock_pc
            store.index = Mock()
            return store
    
    @pytest.mark.asyncio
    async def test_initialize(self, vector_store):
        """Test vector store initialization"""
        with patch('pinecone.Pinecone') as mock_pc_class:
            mock_pc = Mock()
            mock_pc_class.return_value = mock_pc
            mock_pc.list_indexes.return_value = []
            
            vector_store.pc = mock_pc
            
            await vector_store.initialize()
            # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_health_check(self, vector_store):
        """Test health check"""
        vector_store.index.describe_index_stats.return_value = {"dimension": 384}
        
        result = await vector_store.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_store_document_chunks(self, vector_store):
        """Test storing document chunks"""
        chunks = [
            {
                "heading": "Introduction",
                "text": "This is the introduction.",
                "images": []
            },
            {
                "heading": "Chapter 1",
                "text": "This is chapter 1 content.",
                "images": ["image1.png"]
            }
        ]
        
        with patch.object(vector_store.embedding_service, 'encode_text', 
                         return_value=np.random.rand(384)):
            
            document_id = await vector_store.store_document_chunks(
                "test.pdf", chunks, "test_hash"
            )
            
            assert isinstance(document_id, str)
            vector_store.index.upsert.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_pdf(self, vector_store):
        """Test searching PDF"""
        mock_matches = [
            Mock(
                id="test_chunk_1",
                score=0.9,
                metadata={
                    "heading": "Introduction",
                    "text": "Test content",
                    "images": "[]",
                    "chunk_index": 0
                }
            )
        ]
        
        vector_store.index.query.return_value = Mock(matches=mock_matches)
        
        with patch.object(vector_store.embedding_service, 'encode_text',
                         return_value=np.random.rand(384)):
            
            results = await vector_store.search_pdf("test.pdf", "test query")
            
            assert results["total_matches"] == 1
            assert len(results["matches"]) == 1
            assert results["matches"][0]["heading"] == "Introduction"
    
    @pytest.mark.asyncio
    async def test_list_processed_pdfs(self, vector_store):
        """Test listing processed PDFs"""
        mock_stats = Mock()
        mock_stats.namespaces = {
            "pdf_document1": {"vector_count": 10},
            "pdf_document2": {"vector_count": 15}
        }
        
        vector_store.index.describe_index_stats.return_value = mock_stats
        
        pdfs = await vector_store.list_processed_pdfs()
        
        assert len(pdfs) == 2
        assert "document1.pdf" in pdfs
        assert "document2.pdf" in pdfs

if __name__ == "__main__":
    pytest.main([__file__])