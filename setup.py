#!/usr/bin/env python3
"""
Simple setup script for RAG PDF Processing API
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    print("🚀 Setting up RAG PDF Processing API...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install Minieu and core dependencies
    if not run_command(
        'pip3 install "mineru[core]==2.1.0"',
        "Installing Minieu with core dependencies"
    ):
        sys.exit(1)
    
    # Install additional dependencies
    if not run_command(
        'pip3 install "huggingface_hub>=0.33.5" sentence-transformers==2.2.2 chromadb==1.0.16 pdf2image==1.17.0 PyMuPDF==1.26.3',
        "Installing additional dependencies"
    ):
        sys.exit(1)
    
    # Install FastAPI and other requirements
    if not run_command(
        'pip3 install -r requirements.txt',
        "Installing FastAPI and other requirements"
    ):
        sys.exit(1)
    
    # Create necessary directories
    directories = ['uploads', 'minieu_output', 'chroma_db', 'output', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy environment template
    if os.path.exists('env.example') and not os.path.exists('.env'):
        import shutil
        shutil.copy('env.example', '.env')
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your OpenAI API key")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run: python run.py")
    print("3. Open http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()
