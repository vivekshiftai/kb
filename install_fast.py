#!/usr/bin/env python3
"""
Fast installation script for KB PDF Processing API
Installs only the essential dependencies needed for the application to run.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("🚀 Fast Installation for KB PDF Processing API")
    print("=" * 50)
    
    # Step 1: Install core FastAPI dependencies (fastest)
    core_deps = [
        "fastapi==0.116.1",
        "uvicorn[standard]==0.24.0", 
        "python-multipart==0.0.20",
        "pydantic==2.5.0",
        "pydantic-settings==2.1.0",
        "python-dotenv==1.0.0",
        "aiofiles==23.2.1",
        "starlette==0.27.0"
    ]
    
    if not run_command(f"pip install {' '.join(core_deps)}", "Installing core FastAPI dependencies"):
        return False
    
    # Step 2: Install PDF processing (medium speed)
    pdf_deps = [
        "PyMuPDF==1.26.3",
        "pdf2image==1.17.0", 
        "Pillow==11.3.0"
    ]
    
    if not run_command(f"pip install {' '.join(pdf_deps)}", "Installing PDF processing dependencies"):
        return False
    
    # Step 3: Install MinerU (can be slow)
    if not run_command("pip install 'mineru[core]==2.1.0'", "Installing MinerU"):
        return False
    
    # Step 4: Install vector database and embeddings (slowest)
    vector_deps = [
        "chromadb==1.0.16",
        "sentence-transformers==2.2.2",
        "huggingface_hub>=0.33.5"
    ]
    
    if not run_command(f"pip install {' '.join(vector_deps)}", "Installing vector database and embeddings"):
        return False
    
    # Step 5: Install OpenAI (fast)
    if not run_command("pip install openai==1.3.7", "Installing OpenAI client"):
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n📝 Next steps:")
    print("1. Copy env.example to .env: cp env.example .env")
    print("2. Edit .env with your OpenAI API key")
    print("3. Create directories: mkdir -p uploads minieu_output chroma_db output logs")
    print("4. Run the application: python run.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
