#!/usr/bin/env python3
"""
Test script to verify the Minieu/MinerU spelling fix
"""

import os
import sys
from config.settings import get_settings

def test_settings():
    """Test that settings are loaded correctly"""
    try:
        settings = get_settings()
        print(f"✅ Settings loaded successfully")
        print(f"MINERU_OUTPUT_DIR: {settings.MINERU_OUTPUT_DIR}")
        print(f"Directory exists: {os.path.exists(settings.MINERU_OUTPUT_DIR)}")
        
        # Test that the directory can be created
        os.makedirs(settings.MINERU_OUTPUT_DIR, exist_ok=True)
        print(f"✅ Directory created/verified: {settings.MINERU_OUTPUT_DIR}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False

def test_imports():
    """Test that all modules import correctly"""
    try:
        from services.minieu_processor import MinieuProcessor
        from services.pdf_processor import PDFProcessor
        print("✅ All service imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Minieu/MinerU spelling fix...")
    
    success = True
    success &= test_settings()
    success &= test_imports()
    
    if success:
        print("🎉 All tests passed! The spelling fix is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
