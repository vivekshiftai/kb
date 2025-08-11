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
            logger.info(f"📄 Starting PDF processing for {filename}")
            processing_start = datetime.now()
            
            # Extract text and metadata
            logger.info("📖 Opening PDF document...")
            doc = fitz.open(pdf_path)
            logger.info(f"✅ PDF opened successfully", total_pages=doc.page_count)
            
            # Extract metadata
            logger.info("📋 Extracting PDF metadata...")
            metadata = self._extract_metadata(doc, pdf_path)
            logger.info("✅ Metadata extraction completed", 
                       title=metadata.get("title", "N/A"),
                       author=metadata.get("author", "N/A"),
                       page_count=metadata.get("page_count", 0))
            
            # Extract text content
            logger.info("📝 Extracting text content from pages...")
            text_content = self._extract_text_content(doc)
            logger.info(f"✅ Text extraction completed", pages_processed=len(text_content))
            
            # Extract images
            logger.info("🖼️ Extracting images from PDF...")
            images = await self._extract_images(doc, filename)
            logger.info(f"✅ Image extraction completed", images_found=len(images))
            
            # Create chunks
            logger.info("🔧 Creating content chunks...")
            chunks = self._create_chunks(text_content, images)
            logger.info(f"✅ Chunking completed", total_chunks=len(chunks))
            
            doc.close()
            logger.info("✅ PDF document closed")
            
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            result = {
                "filename": filename,
                "metadata": metadata,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_images": len(images),
                "processing_date": datetime.now().isoformat()
            }
            
            logger.info(f"🎉 PDF processing completed successfully for {filename}", 
                       total_chunks=len(chunks), 
                       total_images=len(images),
                       processing_time=f"{processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF {filename}: {e}")
            import traceback
            logger.error("❌ Full error traceback:", traceback=traceback.format_exc())
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
                        
                        # Handle different color formats
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Can save directly as PNG
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
                        else:
                            # CMYK or other formats - convert to RGB
                            try:
                                # Convert to RGB
                                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                                img_filename = f"page_{page_num + 1}_img_{img_index + 1}.png"
                                img_path = os.path.join(output_dir, img_filename)
                                rgb_pix.save(img_path)
                                
                                images.append({
                                    "filename": img_filename,
                                    "path": img_path,
                                    "page_number": page_num + 1,
                                    "image_index": img_index,
                                    "width": rgb_pix.width,
                                    "height": rgb_pix.height
                                })
                                
                                rgb_pix = None
                            except Exception as conv_error:
                                logger.warning(f"Failed to convert image {img_index} from page {page_num + 1}: {conv_error}")
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
            
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []

    def _create_chunks(self, pages_content: List[Dict[str, Any]], 
                      images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create text chunks using PDF ToC titles for heading detection"""
        try:
            chunks = []
            
            # Extract PDF ToC titles for heading detection
            pdf_titles = self._extract_pdf_toc_titles()
            
            for page_content in pages_content:
                page_num = page_content["page_number"]
                structured_content = page_content["structured_content"]
                
                # Group content by headings using PDF ToC titles
                current_heading = "Introduction"
                current_text = []
                page_images = [img for img in images if img["page_number"] == page_num]
                
                for content in structured_content:
                    # Check if content is a heading using PDF ToC titles
                    if self._is_heading_by_toc(content["text"], pdf_titles):
                        # Save previous chunk if it has content
                        if current_text:
                            text = " ".join(current_text)
                            if text.strip():
                                chunks.append({
                                    "heading": current_heading,
                                    "text": text,
                                    "page_number": page_num,
                                    "chunk_index": len(chunks),
                                    "images": [img["path"] for img in page_images],
                                    "tables": []  # Will be populated if tables are found
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
                        chunks.append({
                            "heading": current_heading,
                            "text": text,
                            "page_number": page_num,
                            "chunk_index": len(chunks),
                            "images": [img["path"] for img in page_images],
                            "tables": []
                        })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []
    
    def _extract_pdf_toc_titles(self) -> set:
        """Extract PDF table of contents titles for heading detection"""
        try:
            toc = self.doc.get_toc()
            pdf_titles = set(entry[1].strip() for entry in toc if entry[1].strip())
            logger.info(f"Extracted {len(pdf_titles)} PDF ToC titles for heading detection")
            return pdf_titles
        except Exception as e:
            logger.warning(f"Error extracting PDF ToC titles: {e}")
            return set()
    
    def _is_heading_by_toc(self, text: str, pdf_titles: set) -> bool:
        """Check if text is a heading by comparing with PDF ToC titles"""
        import string
        
        def normalize_line(line):
            """Normalize line for comparison"""
            import re
            line = re.sub(r'[\r\n\t]', ' ', line)
            line = line.lower()
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = re.sub(r'\s+', ' ', line)
            return line.strip()
        
        normalized_text = normalize_line(text)
        
        for title in pdf_titles:
            if normalized_text == normalize_line(title):
                return True
        
        return False