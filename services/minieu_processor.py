import os
import subprocess
import asyncio
import logging
import time
from typing import Dict, Any
from datetime import datetime

from config.settings import get_settings

logger = logging.getLogger(__name__)

class MinieuProcessor:
    """Service to handle Minieu PDF processing"""
    
    def __init__(self):
        self.settings = get_settings()
        
    async def process_pdf_with_minieu(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Process PDF using Minieu and wait for completion"""
        try:
            logger.info(f"🤖 Starting Minieu processing for {filename} - Path: {pdf_path}, Step: minieu_start")
            
            processing_start = datetime.now()
            
            # Get PDF name without extension
            pdf_name = os.path.splitext(filename)[0]
            
            # Check if MinerU output already exists
            minieu_output_dir = os.path.join(self.settings.MINERU_OUTPUT_DIR, pdf_name)
            if os.path.exists(minieu_output_dir):
                logger.info(f"✅ Minieu output already exists for {filename} - Dir: {minieu_output_dir}, Step: minieu_already_processed")
                return {
                    "success": True,
                    "message": "Minieu output already exists",
                    "output_dir": minieu_output_dir,
                    "processing_time": 0
                }
            
            # Create MinerU output directory
            os.makedirs(self.settings.MINERU_OUTPUT_DIR, exist_ok=True)
            
            # Call Minieu to process the PDF
            logger.info(f"🚀 Calling Minieu to process {filename}... - Step: minieu_call")
            
            # MinerU v2.1.0 command: mineru process -p <pdf_path> --output <output_dir>
            # The new version requires -p/--path option
            cmd = [
                "mineru", "process", 
                "-p", pdf_path,
                "--output", self.settings.MINERU_OUTPUT_DIR
            ]
            
            # Alternative command format with --path
            fallback_cmd = [
                "mineru", "process", 
                "--path", pdf_path,
                "--output", self.settings.MINERU_OUTPUT_DIR
            ]
            
            logger.info(f"📋 Minieu command: {' '.join(cmd)}")
            logger.info(f"⏱️ Starting MinerU processing - this may take several minutes for large PDFs...")
            
            # Run MinerU process with real-time output logging
            try:
                logger.info(f"🔄 Starting MinerU processing with real-time output...")
                
                # First try primary command with real-time output
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT  # Redirect stderr to stdout for combined output
                )
                
                # Read output in real-time without timeout
                output_lines = []
                line_count = 0
                start_time = time.time()
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode().strip()
                    if line_str:
                        logger.info(f"📋 MinerU: {line_str}")
                        output_lines.append(line_str)
                        line_count += 1
                        
                        # Show progress every 10 lines or every 30 seconds
                        if line_count % 10 == 0 or (time.time() - start_time) > 30:
                            elapsed = time.time() - start_time
                            logger.info(f"📊 MinerU Progress: {line_count} output lines, {elapsed:.1f}s elapsed")
                            start_time = time.time()
                
                await process.wait()
                
                if process.returncode != 0:
                    logger.warning(f"⚠️ Primary MinerU command failed, trying fallback for {filename} - Return code: {process.returncode}, Step: minieu_fallback")
                    
                    # Try fallback command with real-time output
                    process = await asyncio.create_subprocess_exec(
                        *fallback_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT
                    )
                    
                    # Read fallback output in real-time without timeout
                    output_lines = []
                    line_count = 0
                    start_time = time.time()
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        line_str = line.decode().strip()
                        if line_str:
                            logger.info(f"📋 MinerU (fallback): {line_str}")
                            output_lines.append(line_str)
                            line_count += 1
                            
                            # Show progress every 10 lines or every 30 seconds
                            if line_count % 10 == 0 or (time.time() - start_time) > 30:
                                elapsed = time.time() - start_time
                                logger.info(f"📊 MinerU Fallback Progress: {line_count} output lines, {elapsed:.1f}s elapsed")
                                start_time = time.time()
                    
                    await process.wait()
                    
                    if process.returncode != 0:
                        logger.error(f"❌ MinerU processing failed for {filename} - Return code: {process.returncode}, Step: minieu_failed")
                        raise Exception(f"MinerU processing failed with return code: {process.returncode}")
                else:
                    logger.info(f"✅ Primary MinerU command succeeded for {filename}")
                    
            except Exception as e:
                logger.error(f"❌ Error running MinerU process for {filename}: {str(e)}")
                raise
            
            logger.info(f"✅ Minieu processing completed for {filename} - Total output lines: {len(output_lines)}, Step: minieu_completed")
            
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
            
            logger.info(f"🎉 Minieu processing successful for {filename} - Output: {minieu_output_dir}, Auto: {auto_dir}, Time: {processing_time:.2f}s, Step: minieu_success")
            
            return {
                "success": True,
                "message": "Minieu processing completed successfully",
                "output_dir": minieu_output_dir,
                "auto_dir": auto_dir,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error in Minieu processing for {filename} - Error: {str(e)}, Step: minieu_error")
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
        """Check if MinerU is available in the system"""
        try:
            result = subprocess.run(
                ["mineru", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"✅ MinerU is available: {version_info}")
                
                # Check if it's v2.1.0 or higher
                if "2.1.0" in version_info or "2." in version_info:
                    logger.info("✅ MinerU v2.x detected - using enhanced features")
                else:
                    logger.warning("⚠️ MinerU v1.x detected - some features may not be available")
                
                return True
            else:
                logger.warning(f"⚠️ MinerU not available: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking MinerU availability: {e}")
            return False
    
    def get_minieu_version(self) -> str:
        """Get MinerU version string"""
        try:
            result = subprocess.run(["mineru", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
        except Exception as e:
            logger.error(f"❌ Error getting MinerU version: {e}")
            return "Error"
