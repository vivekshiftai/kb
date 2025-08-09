import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
import tempfile

from main import app
from config.settings import get_settings

client = TestClient(app)

@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = get_settings()
    settings.UPLOAD_DIR = tempfile.mkdtemp()
    settings.OUTPUT_DIR = tempfile.mkdtemp()
    return settings

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    # This would be actual PDF bytes in a real test
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        with patch('services.vector_store.VectorStore.health_check', return_value=True), \
             patch('services.openai_client.OpenAIClient.check_connection', return_value=True):
            
            response = client.get("/health/")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["vector_store_status"] == "connected"
            assert data["openai_status"] == "connected"

    def test_health_check_degraded(self):
        """Test degraded health check"""
        with patch('services.vector_store.VectorStore.health_check', return_value=False), \
             patch('services.openai_client.OpenAIClient.check_connection', return_value=True):
            
            response = client.get("/health/")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["vector_store_status"] == "error"

class TestPDFListEndpoint:
    """Test PDF listing endpoint"""
    
    def test_list_empty_pdfs(self):
        """Test listing when no PDFs are processed"""
        with patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=[]):
            response = client.get("/pdfs/")
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 0
            assert data["pdfs"] == []

    def test_list_pdfs_with_data(self):
        """Test listing PDFs with data"""
        mock_pdfs = ["document1.pdf", "document2.pdf"]
        mock_stats = {"vector_count": 10}
        
        with patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=mock_pdfs), \
             patch('services.vector_store.VectorStore.get_pdf_stats', return_value=mock_stats):
            
            response = client.get("/pdfs/")
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 2
            assert len(data["pdfs"]) == 2
            assert data["pdfs"][0]["filename"] == "document1.pdf"
            assert data["pdfs"][0]["chunk_count"] == 10

class TestQueryEndpoint:
    """Test PDF query endpoint"""
    
    def test_query_nonexistent_pdf(self):
        """Test querying a PDF that doesn't exist"""
        with patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=[]):
            response = client.post("/query/", json={
                "pdf_filename": "nonexistent.pdf",
                "query": "What is this about?"
            })
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_query_existing_pdf(self):
        """Test querying an existing PDF"""
        mock_pdfs = ["test.pdf"]
        mock_search_results = {
            "matches": [{
                "id": "test_chunk_1",
                "score": 0.9,
                "heading": "Introduction",
                "text": "This is a test document about testing.",
                "images": []
            }],
            "total_matches": 1
        }
        mock_openai_response = {
            "answer": "This document is about testing procedures.",
            "confidence": 0.8
        }
        
        with patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=mock_pdfs), \
             patch('services.vector_store.VectorStore.search_pdf', return_value=mock_search_results), \
             patch('services.openai_client.OpenAIClient.generate_response', return_value=mock_openai_response):
            
            response = client.post("/query/", json={
                "pdf_filename": "test.pdf",
                "query": "What is this about?"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["pdf_filename"] == "test.pdf"
            assert data["answer"] == "This document is about testing procedures."
            assert data["total_matches"] == 1
            assert len(data["results"]) == 1

    def test_query_no_results(self):
        """Test query with no matching results"""
        mock_pdfs = ["test.pdf"]
        mock_search_results = {"matches": [], "total_matches": 0}
        
        with patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=mock_pdfs), \
             patch('services.vector_store.VectorStore.search_pdf', return_value=mock_search_results):
            
            response = client.post("/query/", json={
                "pdf_filename": "test.pdf",
                "query": "What is this about?"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["total_matches"] == 0
            assert "couldn't find any relevant information" in data["answer"]

class TestUploadEndpoint:
    """Test PDF upload endpoint"""
    
    def test_upload_invalid_file_type(self):
        """Test uploading non-PDF file"""
        response = client.post(
            "/upload-pdf/",
            files={"file": ("test.txt", b"This is not a PDF", "text/plain")}
        )
        assert response.status_code == 400
        assert "Only PDF files are allowed" in response.json()["detail"]

    def test_upload_valid_pdf(self, sample_pdf_content):
        """Test uploading valid PDF"""
        with patch('utils.helpers.validate_pdf_file', return_value={"valid": True, "page_count": 1}), \
             patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=[]), \
             patch('utils.file_utils.get_file_hash', return_value="test_hash"):
            
            response = client.post(
                "/upload-pdf/",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["pdf_filename"] == "test.pdf"
            assert data["processing_status"] == "processing"

    def test_upload_already_processed_pdf(self, sample_pdf_content):
        """Test uploading PDF that's already processed"""
        with patch('utils.helpers.validate_pdf_file', return_value={"valid": True, "page_count": 1}), \
             patch('services.vector_store.VectorStore.list_processed_pdfs', return_value=["test.pdf"]):
            
            response = client.post(
                "/upload-pdf/",
                files={"file": ("test.pdf", sample_pdf_content, "application/pdf")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["processing_status"] == "completed"
            assert "already processed" in data["message"]

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG PDF Processing API"
        assert data["version"] == "2.0.0"
        assert "endpoints" in data

if __name__ == "__main__":
    pytest.main([__file__])