import os
import json
import logging
from typing import Dict, List, Any
import fitz  # PyMuPDF
import re
from datetime import datetime

from config.settings import get_settings
from utils.helpers import chunk_text_with_overlap, extract_images_from_pdf

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF processing service"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def process_pdf(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Process PDF and extract text, images, and metadata"""
        try:
            logger.info(f"Starting PDF processing for {filename}")
            
            # Extract text and metadata
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = self._extract_metadata(doc, pdf_path)
            
            # Extract text content
            text_content = self._extract_text_content(doc)
            
            # Extract images
            images = await self._extract_images(doc, filename)
            
            # Create chunks
            chunks = self._create_chunks(text_content, images)
            
            doc.close()
            
            result = {
                "filename": filename,
                "metadata": metadata,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_images": len(images),
                "processing_date": datetime.now().isoformat()
            }
            
            logger.info(f"PDF processing completed for {filename}: {len(chunks)} chunks, {len(images)} images")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise

    def _extract_metadata(self, doc: fitz.Document, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            metadata = doc.metadata
            
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_path)
            }
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {e}")
            return {"page_count": 0, "file_size": 0}

    def _extract_text_content(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract text content from PDF pages"""
        try:
            pages_content = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Extract text blocks with formatting info
                blocks = page.get_text("dict")
                
                # Process blocks to identify headings and structure
                structured_content = self._process_text_blocks(blocks, page_num)
                
                pages_content.append({
                    "page_number": page_num + 1,
                    "raw_text": text,
                    "structured_content": structured_content
                })
            
            return pages_content
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return []

    def _process_text_blocks(self, blocks_dict: Dict, page_num: int) -> List[Dict[str, Any]]:
        """Process text blocks to identify structure"""
        try:
            structured_content = []
            
            for block in blocks_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        # Determine if this is likely a heading based on font size and formatting
                        font_size = span.get("size", 12)
                        font_flags = span.get("flags", 0)
                        is_bold = bool(font_flags & 2**4)
                        
                        content_type = "heading" if (font_size > 14 or is_bold) and len(text) < 100 else "text"
                        
                        structured_content.append({
                            "text": text,
                            "type": content_type,
                            "font_size": font_size,
                            "is_bold": is_bold,
                            "page_number": page_num + 1,
                            "bbox": span.get("bbox", [])
                        })
            
            return structured_content
            
        except Exception as e:
            logger.error(f"Error processing text blocks: {e}")
            return []

    async def _extract_images(self, doc: fitz.Document, filename: str) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        try:
            images = []
            output_dir = os.path.join(self.settings.OUTPUT_DIR, "images", 
                                    os.path.splitext(filename)[0])
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
                            
                            images.append({
                                "filename": img_filename,
                                "path": img_path,
                                "page_number": page_num + 1,
                                "image_index": img_index,
                                "width": pix.width,
                                "height": pix.height
                            })
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
            
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []

    def _create_chunks(self, pages_content: List[Dict[str, Any]], 
                      images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create text chunks with associated images"""
        try:
            chunks = []
            current_heading = "Introduction"
            
            for page_content in pages_content:
                page_num = page_content["page_number"]
                structured_content = page_content["structured_content"]
                
                # Group content by headings
                current_text = []
                page_images = [img for img in images if img["page_number"] == page_num]
                
                for content in structured_content:
                    if content["type"] == "heading" and len(content["text"]) > 5:
                        # Save previous chunk if it has content
                        if current_text:
                            text = " ".join(current_text)
                            if text.strip():
                                # Create chunks from the text
                                text_chunks = chunk_text_with_overlap(
                                    text, 
                                    self.settings.CHUNK_MAX_LENGTH,
                                    self.settings.CHUNK_OVERLAP
                                )
                                
                                for i, chunk_text in enumerate(text_chunks):
                                    chunks.append({
                                        "heading": current_heading,
                                        "text": chunk_text,
                                        "page_number": page_num,
                                        "chunk_index": len(chunks),
                                        "images": [img["path"] for img in page_images] if i == 0 else []
                                    })
                        
                        # Start new section
                        current_heading = content["text"]
                        current_text = []
                    else:
                        current_text.append(content["text"])
                
                # Handle remaining text
                if current_text:
                    text = " ".join(current_text)
                    if text.strip():
                        text_chunks = chunk_text_with_overlap(
                            text,
                            self.settings.CHUNK_MAX_LENGTH,
                            self.settings.CHUNK_OVERLAP
                        )
                        
                        for i, chunk_text in enumerate(text_chunks):
                            chunks.append({
                                "heading": current_heading,
                                "text": chunk_text,
                                "page_number": page_num,
                                "chunk_index": len(chunks),
                                "images": [img["path"] for img in page_images] if i == 0 else []
                            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []