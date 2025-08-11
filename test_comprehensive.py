#!/usr/bin/env python3
"""
Comprehensive test script to check all application components
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_directories():
    """Test if required directories exist"""
    print("📁 Testing directory structure...")
    
    required_dirs = ['uploads', 'outputs', 'chroma_db']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
            print(f"❌ Missing directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    if missing_dirs:
        print(f"🔧 Creating missing directories: {missing_dirs}")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
    
    return len(missing_dirs) == 0

def test_imports():
    """Test all imports"""
    print("📦 Testing imports...")
    
    try:
        from main import app
        print("✅ Main app imported")
        
        from services.pdf_processor import PDFProcessor
        print("✅ PDF processor imported")
        
        from services.vector_store import VectorStore
        print("✅ Vector store imported")
        
        from services.rules_generator import RulesGenerator
        print("✅ Rules generator imported")
        
        from services.embeddings import EmbeddingService
        print("✅ Embedding service imported")
        
        from services.openai_client import OpenAIClient
        print("✅ OpenAI client imported")
        
        from config.settings import get_settings
        print("✅ Settings imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_initialization():
    """Test service initialization"""
    print("🔧 Testing service initialization...")
    
    try:
        from services.pdf_processor import PDFProcessor
        from services.vector_store import VectorStore
        from services.rules_generator import RulesGenerator
        from services.embeddings import EmbeddingService
        from services.openai_client import OpenAIClient
        from config.settings import get_settings
        
        # Test settings
        settings = get_settings()
        print("✅ Settings loaded")
        
        # Test services
        pdf_processor = PDFProcessor()
        print("✅ PDF processor initialized")
        
        embedding_service = EmbeddingService()
        print("✅ Embedding service initialized")
        
        openai_client = OpenAIClient()
        print("✅ OpenAI client initialized")
        
        rules_generator = RulesGenerator()
        print("✅ Rules generator initialized")
        
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vector_store_initialization():
    """Test vector store initialization"""
    print("🗄️ Testing vector store initialization...")
    
    try:
        from services.vector_store import VectorStore
        
        vector_store = VectorStore()
        await vector_store.initialize()
        print("✅ Vector store initialized successfully")
        
        # Test health check
        health = await vector_store.health_check()
        print(f"✅ Vector store health check: {health}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging():
    """Test logging configuration"""
    print("📝 Testing logging configuration...")
    
    try:
        import structlog
        logger = structlog.get_logger(__name__)
        
        # Test basic logging
        logger.info("✅ Basic logging works")
        logger.warning("✅ Warning logging works")
        logger.error("✅ Error logging works")
        
        # Test logging with keyword arguments
        logger.info("✅ Keyword argument logging works", 
                   test_param="value", 
                   number=42)
        
        print("✅ Logging configuration works")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_processing():
    """Test PDF processing capabilities"""
    print("📄 Testing PDF processing capabilities...")
    
    try:
        from services.pdf_processor import PDFProcessor
        import fitz
        
        pdf_processor = PDFProcessor()
        print("✅ PDF processor ready")
        
        # Test if PyMuPDF is working
        print("✅ PyMuPDF (fitz) is available")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting comprehensive application test...")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Imports", test_imports),
        ("Service Initialization", test_service_initialization),
        ("Logging Configuration", test_logging),
        ("PDF Processing", test_pdf_processing),
        ("Vector Store Initialization", test_vector_store_initialization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Application is ready.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
