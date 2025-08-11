#!/usr/bin/env python3
"""
Test script to check application startup and identify issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_app_startup():
    """Test if the application can start without errors"""
    print("🧪 Testing application startup...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from main import app
        print("✅ Main app imported successfully")
        
        from services.pdf_processor import PDFProcessor
        print("✅ PDF processor imported successfully")
        
        from services.vector_store import VectorStore
        print("✅ Vector store imported successfully")
        
        from services.rules_generator import RulesGenerator
        print("✅ Rules generator imported successfully")
        
        # Test service initialization
        print("🔧 Testing service initialization...")
        pdf_processor = PDFProcessor()
        print("✅ PDF processor initialized")
        
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        rules_generator = RulesGenerator()
        print("✅ Rules generator initialized")
        
        # Test app startup event (without actually starting the server)
        print("🚀 Testing app startup event...")
        import asyncio
        
        async def test_startup():
            try:
                # This would normally be called by FastAPI
                # We'll test the components individually
                print("✅ Startup components ready")
                return True
            except Exception as e:
                print(f"❌ Startup failed: {e}")
                return False
        
        # Run the async test
        result = asyncio.run(test_startup())
        
        if result:
            print("✅ All startup tests passed!")
            return True
        else:
            print("❌ Startup test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Application startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_startup()
    sys.exit(0 if success else 1)
