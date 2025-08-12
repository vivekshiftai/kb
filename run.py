#!/usr/bin/env python3
"""
Simple run script for RAG PDF Processing API
"""

import uvicorn
import os
import sys

def main():
    """Run the FastAPI application"""
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("MINIEU_OUTPUT_DIR=./minieu_output")
        print("UPLOAD_DIR=./uploads")
        sys.exit(1)
    
    # Check if required directories exist
    required_dirs = ['uploads', 'minieu_output', 'chroma_db', 'output', 'logs']
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    print("🚀 Starting RAG PDF Processing API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health/")
    print("⏹️  Press Ctrl+C to stop")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
