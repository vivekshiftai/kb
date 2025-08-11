"""
Utility helper functions for text processing and file operations
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional
import unicodedata
from PyPDF2 import PdfReader
from PIL import Image
import io
import structlog


logger = structlog.get_logger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    try:
        # Replace tabs/newlines with space
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Error normalizing text: {e}")
        return text.strip() if text else ""


def chunk_text_with_overlap(text: str, max_length: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks with smart boundary detection
    
    Args:
        text: Text to chunk
        max_length: Maximum chunk length
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        if not text or len(text) <= max_length:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + max_length
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if chunk and len(chunk) > 10:  # Minimum chunk size
                    chunks.append(chunk)
                break
            
            # Try to break at sentence boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings near the end
            sentence_endings = ['.', '!', '?', '\n']
            best_break = -1
            
            # Search backwards from end for good break point
            for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                if chunk_text[i] in sentence_endings and i < len(chunk_text) - 1:
                    # Check if next character suggests end of sentence
                    if i + 1 < len(chunk_text) and (chunk_text[i + 1].isspace() or chunk_text[i + 1].isupper()):
                        best_break = i + 1
                        break
            
            if best_break > 0:
                # Found good sentence boundary
                chunk = text[start:start + best_break].strip()
                if chunk and len(chunk) > 10:
                    chunks.append(chunk)
                start = start + best_break - overlap
            else:
                # No good break found, try word boundary
                words = chunk_text.split()
                if len(words) > 1:
                    # Remove last word to avoid cutting mid-word
                    chunk_text = ' '.join(words[:-1])
                    chunk = chunk_text.strip()
                    if chunk and len(chunk) > 10:
                        chunks.append(chunk)
                    start = start + len(chunk_text) - overlap
                else:
                    # Single long word, just split it
                    chunk = chunk_text.strip()
                    if chunk and len(chunk) > 10:
                        chunks.append(chunk)
                    start = end - overlap
            
            # Ensure we make progress
            if start <= (len(chunks) - 1) * 10 if chunks else 0:
                start += max_length // 2
        
        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text] if text else []


def validate_pdf_file(file_path: str) -> Dict[str, Any]:
    """
    Validate PDF file structure and readability
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Dictionary with validation results
    """
    try:
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File does not exist"}
        
        if not file_path.lower().endswith('.pdf'):
            return {"valid": False, "error": "File is not a PDF"}
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {"valid": False, "error": "File is empty"}
        
        # Try to open and validate PDF structure
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            page_count = doc.page_count
            
            if page_count == 0:
                doc.close()
                return {"valid": False, "error": "PDF has no pages"}
            
            # Try to read first page to ensure it's not corrupted
            first_page = doc[0]
            text = first_page.get_text()
            
            doc.close()
            
            return {
                "valid": True,
                "page_count": page_count,
                "file_size": file_size,
                "has_text": len(text.strip()) > 0
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Cannot open PDF: {str(e)}"}
            
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    
    try:
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    except (ValueError, OverflowError):
        return f"{size_bytes} B"


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    if not filename:
        return ""
    
    try:
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Remove multiple consecutive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Remove leading/trailing underscores and dots
        filename = filename.strip('_.')
        
        # Ensure filename is not too long (255 is typical filesystem limit)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            max_name_len = 255 - len(ext)
            filename = name[:max_name_len] + ext
        
        # Ensure we have a valid filename
        if not filename or filename in ['.', '..']:
            filename = "document.pdf"
        
        return filename
        
    except Exception as e:
        logger.error(f"Error cleaning filename: {e}")
        return "document.pdf"


def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extract features from text for better processing
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of text features
    """
    try:
        if not text:
            return {"word_count": 0, "char_count": 0, "has_numbers": False, "has_special_chars": False}
        
        word_count = len(text.split())
        char_count = len(text)
        has_numbers = bool(re.search(r'\d', text))
        has_special_chars = bool(re.search(r'[^\w\s]', text))
        
        # Detect if text might be a heading (short, title case, etc.)
        is_likely_heading = (
            word_count <= 10 and 
            char_count <= 100 and 
            text.istitle() and 
            not text.endswith('.')
        )
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "has_numbers": has_numbers,
            "has_special_chars": has_special_chars,
            "is_likely_heading": is_likely_heading
        }
        
    except Exception as e:
        logger.error(f"Error extracting text features: {e}")
        return {"word_count": 0, "char_count": 0, "has_numbers": False, "has_special_chars": False}


def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of PIL Image objects extracted from PDF.
    """
    images = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if "/XObject" in page["/Resources"]:
            xobjects = page["/Resources"]["/XObject"].get_object()
            for obj in xobjects:
                if xobjects[obj]["/Subtype"] == "/Image":
                    size = (xobjects[obj]["/Width"], xobjects[obj]["/Height"])
                    data = xobjects[obj].get_data()
                    mode = "RGB" if xobjects[obj]["/ColorSpace"] == "/DeviceRGB" else "P"
                    img = Image.frombytes(mode, size, data)
                    images.append(img)
    return images
