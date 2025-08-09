import re
import string
import logging
from typing import List, Dict, Any, Optional
import unicodedata

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    if not text:
        return ""
    
    # Replace tabs/newlines with space
    text = re.sub(r'[\r\n\t]', ' ', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def chunk_text_with_overlap(text: str, max_length: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    try:
        if not text or len(text) <= max_length:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + max_length
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings near the end
            sentence_endings = ['.', '!', '?', '\n']
            best_break = -1
            
            for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                if chunk_text[i] in sentence_endings and i < len(chunk_text) - 1:
                    if chunk_text[i + 1].isspace() or chunk_text[i + 1].isupper():
                        best_break = i + 1
                        break
            
            if best_break > 0:
                chunks.append(text[start:start + best_break].strip())
                start = start + best_break - overlap
            else:
                # No good break found, split at word boundary
                words = chunk_text.split()
                if len(words) > 1:
                    # Remove last word to avoid cutting mid-word
                    chunk_text = ' '.join(words[:-1])
                    chunks.append(chunk_text)
                    start = start + len(chunk_text) - overlap
                else:
                    # Single long word, just split it
                    chunks.append(chunk_text)
                    start = end - overlap
            
            # Ensure we make progress
            if start <= len(chunks) - 1 if chunks else 0:
                start += max_length // 2
        
        # Clean up chunks
        cleaned_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and len(chunk) > 10:  # Minimum chunk size
                cleaned_chunks.append(chunk)
        
        return cleaned_chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text] if text else []

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[str]:
    """Extract images from PDF file"""
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        image_paths = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                        img_path = os.path.join(output_dir, img_filename)
                        pix.save(img_path)
                        image_paths.append(img_path)
                    
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image: {e}")
        
        doc.close()
        return image_paths
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        return []

def validate_pdf_file(file_path: str) -> Dict[str, Any]:
    """Validate PDF file"""
    try:
        import fitz
        
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File does not exist"}
        
        if not file_path.lower().endswith('.pdf'):
            return {"valid": False, "error": "File is not a PDF"}
        
        # Try to open the PDF
        try:
            doc = fitz.open(file_path)
            page_count = doc.page_count
            file_size = os.path.getsize(file_path)
            doc.close()
            
            return {
                "valid": True,
                "page_count": page_count,
                "file_size": file_size
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Cannot open PDF: {str(e)}"}
            
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')
    
    # Ensure filename is not too long
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_len = 255 - len(ext)
        filename = name[:max_name_len] + ext
    
    return filename