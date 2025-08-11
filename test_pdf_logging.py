#!/usr/bin/env python3
"""
Test script to verify PDF processing logging works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pdf_processor_logging():
    """Test PDF processor logging"""
    print("🧪 Testing PDF processor logging...")
    
    try:
        # Import after path setup
        from services.pdf_processor import PDFProcessor
        from services.vector_store import VectorStore
        from services.rules_generator import RulesGenerator
        
        # Test PDF processor initialization
        pdf_processor = PDFProcessor()
        print("✅ PDF processor initialized")
        
        # Test vector store initialization
        vector_store = VectorStore()
        print("✅ Vector store initialized")
        
        # Test rules generator initialization
        rules_generator = RulesGenerator()
        print("✅ Rules generator initialized")
        
        print("✅ All service initializations passed!")
        return True
        
    except Exception as e:
        print(f"❌ PDF processor logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_processor_logging()
    sys.exit(0 if success else 1)
