#!/usr/bin/env python3
"""
Test script to verify settings loading
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_settings():
    """Test that settings can be loaded"""
    try:
        from config.settings import get_settings
        
        print("✅ Import successful")
        
        settings = get_settings()
        print(f"✅ Settings loaded successfully")
        print(f"MINERU_OUTPUT_DIR: {settings.MINERU_OUTPUT_DIR}")
        print(f"UPLOAD_DIR: {settings.UPLOAD_DIR}")
        print(f"OUTPUT_DIR: {settings.OUTPUT_DIR}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing settings loading...")
    success = test_settings()
    
    if success:
        print("🎉 Settings test passed!")
    else:
        print("❌ Settings test failed!")
        sys.exit(1)
