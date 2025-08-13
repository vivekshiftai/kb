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
        print("Creating .env file from template...")
        try:
            import shutil
            shutil.copy('env.example', '.env')
            print("✅ Created .env file from env.example")
            print("⚠️  Please edit .env file and add your OpenAI API key:")
            print("OPENAI_API_KEY=your_api_key_here")
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            print("⚠️  Please manually create a .env file with your OpenAI API key:")
            print("OPENAI_API_KEY=your_api_key_here")
            print("MINERU_OUTPUT_DIR=./mineru_output")
            print("UPLOAD_DIR=./uploads")
            print("⚠️  Continuing without .env file (using defaults)...")
    
    # Check if required directories exist
    required_dirs = ['uploads', 'mineru_output', 'chromadb', 'output', 'logs']
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
