#!/usr/bin/env python3
"""
Diagnostic and fix script for PDF processing issues
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_directories():
    """Create required directories"""
    print("📁 Creating required directories...")
    
    directories = ['uploads', 'outputs', 'chroma_db']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pydantic-settings',
        'structlog',
        'fitz',  # PyMuPDF
        'sentence-transformers',
        'chromadb',
        'openai',
        'Pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment configuration"""
    print("🔧 Checking environment configuration...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        print(f"✅ Settings loaded successfully")
        print(f"📁 Upload directory: {settings.UPLOAD_DIR}")
        print(f"📁 Output directory: {settings.OUTPUT_DIR}")
        print(f"📁 ChromaDB directory: {settings.CHROMA_PERSIST_DIRECTORY}")
        print(f"📏 Max file size: {settings.MAX_FILE_SIZE}")
        
        # Check if directories exist
        for dir_path in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.CHROMA_PERSIST_DIRECTORY]:
            if not os.path.exists(dir_path):
                print(f"⚠️ Directory doesn't exist: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False

def test_logging():
    """Test logging configuration"""
    print("📝 Testing logging configuration...")
    
    try:
        import structlog
        logger = structlog.get_logger(__name__)
        
        # Test basic logging
        logger.info("✅ Logging test successful")
        
        # Test with keyword arguments
        logger.info("✅ Keyword argument logging works", 
                   test_param="value", 
                   number=42)
        
        print("✅ Logging configuration is working")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

async def test_services():
    """Test service initialization"""
    print("🔧 Testing service initialization...")
    
    try:
        from services.pdf_processor import PDFProcessor
        from services.vector_store import VectorStore
        from services.rules_generator import RulesGenerator
        from services.embeddings import EmbeddingService
        from services.openai_client import OpenAIClient
        
        # Test PDF processor
        pdf_processor = PDFProcessor()
        print("✅ PDF processor initialized")
        
        # Test embedding service
        embedding_service = EmbeddingService()
        print("✅ Embedding service initialized")
        
        # Test OpenAI client
        openai_client = OpenAIClient()
        print("✅ OpenAI client initialized")
        
        # Test rules generator
        rules_generator = RulesGenerator()
        print("✅ Rules generator initialized")
        
        # Test vector store (this might take a moment)
        print("🗄️ Initializing vector store...")
        vector_store = VectorStore()
        await vector_store.initialize()
        print("✅ Vector store initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_file_permissions():
    """Check file permissions"""
    print("🔐 Checking file permissions...")
    
    try:
        # Test if we can create files in the current directory
        test_file = "test_permissions.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("✅ File write permissions OK")
        
        # Test upload directory
        test_upload_file = os.path.join("uploads", "test_permissions.tmp")
        with open(test_upload_file, "w") as f:
            f.write("test")
        os.remove(test_upload_file)
        print("✅ Upload directory permissions OK")
        
        return True
        
    except Exception as e:
        print(f"❌ File permission check failed: {e}")
        return False

async def run_diagnostic():
    """Run complete diagnostic"""
    print("🚀 Starting PDF processing diagnostic...")
    print("=" * 60)
    
    checks = [
        ("Directory Creation", create_directories),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Logging", test_logging),
        ("File Permissions", check_file_permissions),
        ("Service Initialization", test_services),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n🧪 Running: {check_name}")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ Check {check_name} failed with exception: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 60)
    print("📊 Diagnostic Results:")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! PDF processing should work correctly.")
        print("\nNext steps:")
        print("1. Start the server: uvicorn main:app --reload")
        print("2. Try uploading a PDF file")
        print("3. Check the logs if any issues occur")
    else:
        print("⚠️ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check your .env file configuration")
        print("3. Ensure sufficient disk space")
        print("4. Check file permissions")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(run_diagnostic())
    sys.exit(0 if success else 1)
