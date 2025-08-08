import re
import string
import logging
from typing import List, Dict, Any, Optional
import unicodedata

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for consistent matching"""
    if not text:
        return ""
    
    # Replace tabs/newlines with space
    text = re.sub(r'[\r\n\t]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def extract_toc_titles(pdf_path: str) -> List[str]:
    """Extract table of contents titles from PDF"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        titles = [entry[1].strip() for entry in toc if entry[1].strip()]
        doc.close()
        return titles
    except Exception as e:
        logger.error(f"Error extracting TOC: {e}")
        return []

def clean_markdown_text(text: str) -> str:
    """Clean markdown text by removing formatting"""
    if not text:
        return ""
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove image references
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Remove bold/italic formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def extract_numbered_steps(text: str) -> List[str]:
    """Extract numbered steps from text"""
    # Pattern for numbered steps (1., 2., etc.)
    step_pattern = r'(?:^|\n)\s*(\d+\.)\s*([^\n]*(?:\n(?!\s*\d+\.)[^\n]*)*)'
    matches = re.findall(step_pattern, text, re.MULTILINE)
    
    if matches:
        return [f"{num} {content.strip()}" for num, content in matches]
    
    # Fallback: split by lines if no numbered steps found
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

def extract_table_data(table_html: str) -> List[List[str]]:
    """Extract data from HTML table"""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return []
        
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(cells)
        
        return rows
        
    except ImportError:
        logger.warning("BeautifulSoup not available for table parsing")
        return []
    except Exception as e:
        logger.error(f"Error parsing table: {e}")
        return []

def format_text_for_display(text: str, max_length: int = 200) -> str:
    """Format text for display with length limit"""
    if not text:
        return ""
    
    # Clean the text
    cleaned = clean_markdown_text(text)
    
    # Limit length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    try:
        # Simple keyword extraction based on word frequency
        words = normalize_text(text).split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'upon',
            'against', 'within', 'throughout', 'beneath', 'alongside', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def validate_text_quality(text: str) -> Dict[str, Any]:
    """Validate text quality and return metrics"""
    try:
        if not text:
            return {"valid": False, "reason": "Empty text"}
        
        # Basic quality checks
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Check for minimum content
        if word_count < 5:
            return {"valid": False, "reason": "Too few words"}
        
        if char_count < 20:
            return {"valid": False, "reason": "Too short"}
        
        # Check for meaningful content (not just special characters)
        alpha_ratio = sum(c.isalnum() for c in text) / len(text)
        if alpha_ratio < 0.3:
            return {"valid": False, "reason": "Too few alphanumeric characters"}
        
        return {
            "valid": True,
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "alpha_ratio": alpha_ratio
        }
        
    except Exception as e:
        logger.error(f"Error validating text quality: {e}")
        return {"valid": False, "reason": f"Validation error: {str(e)}"}

def chunk_text_by_sentences(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Chunk text by sentences with overlap"""
    try:
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, save current chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text] if text else []