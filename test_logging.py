#!/usr/bin/env python3
"""
Test script to verify logging configuration is working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app to trigger logging configuration
from main import app, logger

def test_logging():
    """Test various logging scenarios"""
    print("🧪 Testing logging configuration...")
    
    try:
        # Test basic logging
        logger.info("✅ Basic info logging works")
        logger.warning("⚠️ Warning logging works")
        logger.error("❌ Error logging works")
        
        # Test logging with keyword arguments
        logger.info("📊 Testing keyword arguments", 
                   test_param="value", 
                   number=42, 
                   boolean=True)
        
        # Test logging with complex data
        logger.info("🔧 Testing complex data", 
                   filename="test.pdf",
                   total_pages=100,
                   processing_time=2.5)
        
        print("✅ All logging tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_logging()
    sys.exit(0 if success else 1)
