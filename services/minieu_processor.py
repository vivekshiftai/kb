import os
import subprocess
import asyncio
import structlog
from typing import Dict, Any
from datetime import datetime

from config.settings import get_settings

logger = structlog.get_logger(__name__)

class MinieuProcessor:
    """Service to handle Minieu PDF processing"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def process_pdf_with_minieu(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Process PDF using Minieu and wait for completion"""
        try:
            logger.info(f"🤖 Starting Minieu processing for {filename}", 
                       pdf_path=pdf_path,
                       step="minieu_start")
            
            processing_start = datetime.now()
            
            # Get PDF name without extension
            pdf_name = os.path.splitext(filename)[0]
            
            # Check if Minieu output already exists
            minieu_output_dir = os.path.join(self.settings.MINIEU_OUTPUT_DIR, pdf_name)
            if os.path.exists(minieu_output_dir):
                logger.info(f"✅ Minieu output already exists for {filename}", 
                           minieu_dir=minieu_output_dir,
                           step="minieu_already_processed")
                return {
                    "success": True,
                    "message": "Minieu output already exists",
                    "output_dir": minieu_output_dir,
                    "processing_time": 0
                }
            
            # Create Minieu output directory
            os.makedirs(self.settings.MINIEU_OUTPUT_DIR, exist_ok=True)
            
            # Call Minieu to process the PDF
            logger.info(f"🚀 Calling Minieu to process {filename}...", 
                       step="minieu_call")
            
            # Minieu command: mineru process <pdf_path> --output <output_dir>
            cmd = [
                "mineru", "process", pdf_path, 
                "--output", self.settings.MINIEU_OUTPUT_DIR
            ]
            
            logger.info(f"📋 Minieu command: {' '.join(cmd)}")
            
            # Run Minieu process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"❌ Minieu processing failed for {filename}", 
                           returncode=process.returncode,
                           stderr=stderr.decode() if stderr else "No error output",
                           step="minieu_failed")
                raise Exception(f"Minieu processing failed: {stderr.decode() if stderr else 'Unknown error'}")
            
            logger.info(f"✅ Minieu processing completed for {filename}", 
                       stdout=stdout.decode() if stdout else "No output",
                       step="minieu_completed")
            
            # Wait a bit for files to be written
            await asyncio.sleep(2)
            
            # Verify Minieu output was created
            if not os.path.exists(minieu_output_dir):
                logger.error(f"❌ Minieu output directory not created: {minieu_output_dir}")
                raise Exception(f"Minieu output directory not found: {minieu_output_dir}")
            
            # Check for auto directory
            auto_dir = self._find_auto_directory(minieu_output_dir)
            if not auto_dir:
                logger.error(f"❌ No auto directory found in Minieu output: {minieu_output_dir}")
                raise Exception(f"No auto directory found in Minieu output: {minieu_output_dir}")
            
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            logger.info(f"🎉 Minieu processing successful for {filename}", 
                       output_dir=minieu_output_dir,
                       auto_dir=auto_dir,
                       processing_time=f"{processing_time:.2f}s",
                       step="minieu_success")
            
            return {
                "success": True,
                "message": "Minieu processing completed successfully",
                "output_dir": minieu_output_dir,
                "auto_dir": auto_dir,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error in Minieu processing for {filename}", 
                        error=str(e),
                        step="minieu_error")
            raise
    
    def _find_auto_directory(self, minieu_output_dir: str) -> str:
        """Find the 'auto' directory within Minieu output"""
        try:
            if not os.path.exists(minieu_output_dir):
                return None
            
            items = os.listdir(minieu_output_dir)
            logger.info(f"📁 Found {len(items)} items in Minieu output: {items}")
            
            for item in items:
                item_path = os.path.join(minieu_output_dir, item)
                if os.path.isdir(item_path):
                    auto_path = os.path.join(item_path, "auto")
                    if os.path.exists(auto_path):
                        logger.info(f"✅ Found auto directory: {auto_path}")
                        return auto_path
            
            logger.warning(f"⚠️ No auto directory found in: {minieu_output_dir}")
            return None
        except Exception as e:
            logger.error(f"Error finding auto directory: {e}")
            return None
    
    def check_minieu_availability(self) -> bool:
        """Check if Minieu is available in the system"""
        try:
            result = subprocess.run(
                ["mineru", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✅ Minieu is available: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"⚠️ Minieu not available: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking Minieu availability: {e}")
            return False
