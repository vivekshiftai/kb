#!/usr/bin/env python3
"""
Simple script to start the FastAPI server and check endpoints
"""

import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting RAG PDF Processing API...")
    print("Server will be available at: http://localhost:8000")
    print("Swagger UI will be available at: http://localhost:8000/docs")
    print("ReDoc will be available at: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
