import os
import json
import logging
from typing import Dict, List, Any
import re
from datetime import datetime
import structlog

from config.settings import get_settings
from utils.helpers import chunk_text_with_overlap

logger = structlog.get_logger(__name__)

class PDFProcessor:
    """Enhanced PDF processing service using Minieu output data"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def process_pdf(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Process PDF using Minieu output data (markdown + images)"""
        try:
            logger.info(f"📄 Starting Minieu-based PDF processing for {filename}", 
                       pdf_path=pdf_path,
                       step="processing_start")
            processing_start = datetime.now()
            
            # Get Minieu output directory for this PDF
            pdf_name = os.path.splitext(filename)[0]
            minieu_output_dir = os.path.join(self.settings.MINIEU_OUTPUT_DIR, pdf_name)
            
            if not os.path.exists(minieu_output_dir):
                raise FileNotFoundError(f"Minieu output directory not found: {minieu_output_dir}")
            
            logger.info(f"📁 Found Minieu output directory", 
                       minieu_dir=minieu_output_dir,
                       step="minieu_dir_found")
            
            # Find the "auto" directory within Minieu output
            auto_dir = self._find_auto_directory(minieu_output_dir)
            if not auto_dir:
                raise FileNotFoundError(f"No 'auto' directory found in {minieu_output_dir}")
            
            logger.info(f"📁 Found auto directory", 
                       auto_dir=auto_dir,
                       step="auto_dir_found")
            
            # Extract markdown content from Minieu output
            logger.info("📖 Extracting markdown content from Minieu output...", 
                       filename=filename,
                       step="markdown_extraction")
            markdown_content = self._extract_markdown_content(auto_dir)
            logger.info(f"✅ Markdown extraction completed", 
                       filename=filename,
                       sections_found=len(markdown_content),
                       step="markdown_complete")
            
            # Extract image information from Minieu output
            logger.info("🖼️ Extracting image information from Minieu output...", 
                       filename=filename,
                       step="image_extraction")
            images = self._extract_images_from_minieu(auto_dir, pdf_name)
            logger.info(f"✅ Image extraction completed", 
                       filename=filename,
                       images_found=len(images),
                       step="image_complete")
            
            # Create chunks from markdown content using heading-based chunking
            logger.info("🔧 Creating content chunks from markdown using heading detection...", 
                       filename=filename,
                       step="chunking")
            chunks = self._create_chunks_from_markdown_with_headings(markdown_content, images)
            logger.info(f"✅ Chunking completed", 
                       filename=filename,
                       total_chunks=len(chunks),
                       step="chunking_complete")
            
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            result = {
                "filename": filename,
                "metadata": {
                    "title": filename,
                    "page_count": len(markdown_content),
                    "file_size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0,
                    "processing_method": "minieu",
                    "auto_directory": auto_dir
                },
                "chunks": chunks,
                "total_chunks": len(chunks),
                "total_images": len(images),
                "processing_date": datetime.now().isoformat()
            }
            
            logger.info(f"🎉 Minieu-based PDF processing completed successfully for {filename}", 
                       filename=filename,
                       total_chunks=len(chunks), 
                       total_images=len(images),
                       processing_time=f"{processing_time:.2f}s",
                       step="processing_complete")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF {filename} with Minieu data", 
                        filename=filename,
                        error=str(e),
                        step="processing_error")
            import traceback
            logger.error("❌ Full error traceback:", 
                        filename=filename,
                        traceback=traceback.format_exc())
            raise

    def _find_auto_directory(self, minieu_output_dir: str) -> str:
        """Find the 'auto' directory within Minieu output"""
        try:
            for item in os.listdir(minieu_output_dir):
                item_path = os.path.join(minieu_output_dir, item)
                if os.path.isdir(item_path):
                    auto_path = os.path.join(item_path, "auto")
                    if os.path.exists(auto_path):
                        return auto_path
            
            return None
        except Exception as e:
            logger.error(f"Error finding auto directory: {e}")
            return None

    def _extract_markdown_content(self, auto_dir: str) -> List[Dict[str, Any]]:
        """Extract markdown content from Minieu auto directory"""
        try:
            markdown_files = []
            
            # Look for markdown files in auto directory
            for file in os.listdir(auto_dir):
                if file.lower().endswith('.md') or file.lower().endswith('.markdown'):
                    file_path = os.path.join(auto_dir, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        markdown_files.append({
                            "filename": file,
                            "file_path": file_path,
                            "content": content
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to read markdown file {file}: {e}")
                        continue
            
            if not markdown_files:
                raise FileNotFoundError(f"No markdown files found in {auto_dir}")
            
            logger.info(f"Found {len(markdown_files)} markdown files in auto directory")
            return markdown_files
            
        except Exception as e:
            logger.error(f"Error extracting markdown content: {e}")
            return []

    def _extract_images_from_minieu(self, auto_dir: str, pdf_name: str) -> List[Dict[str, Any]]:
        """Extract image information from Minieu auto directory"""
        try:
            images = []
            
            # Look for images in auto directory
            images_dir = os.path.join(auto_dir, "images")
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        file_path = os.path.join(images_dir, file)
                        
                        # Create relative path for storage
                        rel_path = os.path.relpath(file_path, self.settings.OUTPUT_DIR)
                        
                        images.append({
                            "filename": file,
                            "path": file_path,
                            "relative_path": rel_path,
                            "image_index": len(images) + 1
                        })
            
            logger.info(f"Found {len(images)} images in Minieu auto directory")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images from Minieu output: {e}")
            return []

    def _create_chunks_from_markdown_with_headings(self, markdown_content: List[Dict[str, Any]], 
                                                 images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from markdown content using heading-based detection (#)"""
        try:
            chunks = []
            
            for md_file in markdown_content:
                content = md_file["content"]
                filename = md_file["filename"]
                
                logger.info(f"Processing markdown file: {filename}")
                
                # Split content into lines
                lines = content.split('\n')
                
                current_heading = None
                current_content = []
                current_images = []
                current_tables = []
                
                for line_num, line in enumerate(lines, 1):
                    stripped_line = line.strip()
                    
                    # Check if line is a heading (starts with #)
                    if stripped_line.startswith('#'):
                        # Save previous chunk if it exists
                        if current_heading is not None:
                            text = '\n'.join(current_content).strip()
                            if text:  # Only create chunk if there's content
                                chunks.append({
                                    "heading": current_heading,
                                    "text": text,
                                    "section_number": len(chunks) + 1,
                                    "chunk_index": len(chunks),
                                    "images": current_images.copy(),
                                    "tables": current_tables.copy(),
                                    "source_file": filename,
                                    "line_start": line_num - len(current_content) if current_content else line_num
                                })
                        
                        # Start new chunk
                        current_heading = stripped_line
                        current_content = []
                        current_images = []
                        current_tables = []
                        
                        logger.debug(f"New heading detected: {current_heading}")
                        continue
                    
                    # Check for image references
                    if '![](' in line:
                        # Extract image path from markdown image syntax
                        img_match = re.search(r'!\[\]\((.+?)\)', line)
                        if img_match:
                            img_path = img_match.group(1)
                            # Find corresponding image in images list
                            for img in images:
                                if img["filename"] in img_path or img_path in img["path"]:
                                    current_images.append(img["path"])
                                    break
                            else:
                                # If not found in images list, add the path as is
                                current_images.append(img_path)
                        
                        logger.debug(f"Image reference found: {img_path}")
                        continue
                    
                    # Check for table content
                    if '<table>' in line or '|' in line and '---' in line:
                        current_tables.append(line)
                        logger.debug(f"Table content found")
                        continue
                    
                    # Add to current content
                    if line.strip():  # Only add non-empty lines
                        current_content.append(line)
                
                # Save final chunk
                if current_heading is not None:
                    text = '\n'.join(current_content).strip()
                    if text:  # Only create chunk if there's content
                        chunks.append({
                            "heading": current_heading,
                            "text": text,
                            "section_number": len(chunks) + 1,
                            "chunk_index": len(chunks),
                            "images": current_images.copy(),
                            "tables": current_tables.copy(),
                            "source_file": filename,
                            "line_start": len(lines) - len(current_content) if current_content else len(lines)
                        })
            
            # Log chunk creation summary
            total_chunks = len(chunks)
            chunks_with_images = sum(1 for chunk in chunks if chunk.get("images"))
            total_images = sum(len(chunk.get("images", [])) for chunk in chunks)
            
            logger.info(f"Created {total_chunks} chunks from Minieu markdown using heading detection", 
                       chunks_with_images=chunks_with_images,
                       total_images=total_images)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks from markdown: {e}")
            return []