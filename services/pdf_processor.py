import os
import json
import subprocess
import tempfile
import logging
from typing import Dict, List, Any
import fitz  # PyMuPDF
from PIL import Image
import re
import string

from config.settings import get_settings
from config.mineru_config import create_mineru_config
from utils.helpers import normalize_text, extract_toc_titles

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processing service using MinerU"""
    
    def __init__(self):
        self.settings = get_settings()
        self.config_path = create_mineru_config()
        
    async def process_pdf(self, pdf_path: str, filename: str) -> str:
        """Process PDF with MinerU and return output directory"""
        try:
            logger.info(f"Starting PDF processing for {filename}")
            
            # Create output directory
            output_dir = os.path.join(self.settings.OUTPUT_DIR, 
                                    os.path.splitext(filename)[0])
            os.makedirs(output_dir, exist_ok=True)
            
            # Run MinerU CLI
            cmd = [
                "mineru",
                "-p", pdf_path,
                "-o", output_dir,
                "-m", "auto"
            ]
            
            logger.info(f"Running MinerU command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                raise Exception(f"MinerU failed: {result.stderr}")
            
            logger.info(f"MinerU processing completed for {filename}")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MinerU subprocess error: {e.stderr}")
            raise Exception(f"PDF processing failed: {e.stderr}")
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise

    async def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        try:
            doc = fitz.open(pdf_path)
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

    def extract_toc_titles(self, pdf_path: str) -> List[str]:
        """Extract table of contents titles from PDF"""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            titles = [entry[1].strip() for entry in toc if entry[1].strip()]
            doc.close()
            return titles
        except Exception as e:
            logger.error(f"Error extracting TOC: {e}")
            return []

    def chunk_markdown_with_toc(self, md_path: str, toc_titles: List[str]) -> List[Dict[str, Any]]:
        """Chunk markdown content using PDF TOC titles"""
        try:
            logger.info(f"Chunking markdown file: {md_path}")
            
            heading_pattern = re.compile(r"^#+\s*\d+(?:\.\d+)*\b")
            image_pattern = re.compile(r"!\[\]\((.+?)\)")
            table_pattern = re.compile(r"(<table>.*?</table>)", re.DOTALL)
            
            chunks = []
            current_heading = None
            content_lines = []
            images = []
            tables = []
            
            # Normalize TOC titles for matching
            normalized_toc = {normalize_text(title): title for title in toc_titles}
            
            with open(md_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                stripped = line.strip()
                normalized_line = normalize_text(line)
                
                # Check if this is a heading
                is_heading = False
                if stripped.startswith("#") and heading_pattern.match(stripped):
                    is_heading = True
                elif normalized_line in normalized_toc:
                    is_heading = True
                
                if is_heading:
                    # Save previous chunk
                    if current_heading is not None:
                        text = "".join(content_lines).strip()
                        if text:  # Only add non-empty chunks
                            chunks.append({
                                "heading": current_heading,
                                "text": text,
                                "images": images.copy(),
                                "tables": tables.copy()
                            })
                    
                    # Start new chunk
                    current_heading = stripped
                    content_lines.clear()
                    images.clear()
                    tables.clear()
                    continue
                
                # Check for images
                img_match = image_pattern.search(line)
                if img_match:
                    images.append(img_match.group(1))
                    continue
                
                # Check for tables
                table_matches = table_pattern.findall(line)
                if table_matches:
                    tables.extend(table_matches)
                    continue
                
                # Add to content
                content_lines.append(line)
            
            # Save final chunk
            if current_heading is not None:
                text = "".join(content_lines).strip()
                if text:
                    chunks.append({
                        "heading": current_heading,
                        "text": text,
                        "images": images.copy(),
                        "tables": tables.copy()
                    })
            
            logger.info(f"Created {len(chunks)} chunks from markdown")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking markdown: {e}")
            return []

    def find_markdown_files(self, output_dir: str) -> List[str]:
        """Find markdown files in the output directory"""
        md_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return md_files

    def find_image_files(self, output_dir: str) -> List[str]:
        """Find image files in the output directory"""
        image_files = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(os.path.join(root, file))
        return image_files

    async def get_processing_results(self, output_dir: str, pdf_path: str) -> Dict[str, Any]:
        """Get all processing results from output directory"""
        try:
            # Find markdown and image files
            md_files = self.find_markdown_files(output_dir)
            image_files = self.find_image_files(output_dir)
            
            # Extract TOC from original PDF
            toc_titles = self.extract_toc_titles(pdf_path)
            
            # Process each markdown file
            all_chunks = []
            for md_file in md_files:
                chunks = self.chunk_markdown_with_toc(md_file, toc_titles)
                all_chunks.extend(chunks)
            
            return {
                "chunks": all_chunks,
                "images": image_files,
                "markdown_files": md_files,
                "toc_titles": toc_titles
            }
            
        except Exception as e:
            logger.error(f"Error getting processing results: {e}")
            raise