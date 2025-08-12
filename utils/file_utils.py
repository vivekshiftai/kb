import os
import hashlib
import shutil
import logging
from datetime import datetime, timedelta
from typing import List

from config.settings import get_settings

logger = logging.getLogger(__name__)

def ensure_directories():
    """Create necessary directories if they don't exist"""
    settings = get_settings()
    
    directories = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.MINIEU_OUTPUT_DIR,
        os.path.join(settings.OUTPUT_DIR, "images")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of a file"""
    try:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error generating file hash: {e}")
        return ""

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return 0

def cleanup_old_files(directory: str, days: int = 7):
    """Clean up files older than specified days"""
    try:
        if not os.path.exists(directory):
            return
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_modified < cutoff_date:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"Deleted old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {e}")
            
            elif os.path.isdir(file_path):
                try:
                    # Check if directory is old and empty or contains only old files
                    dir_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if dir_modified < cutoff_date:
                        shutil.rmtree(file_path)
                        deleted_count += 1
                        logger.info(f"Deleted old directory: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete directory {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files/directories from {directory}")
            
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")

def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate if file has allowed extension"""
    if not filename:
        return False
    
    file_extension = os.path.splitext(filename.lower())[1]
    return file_extension in [ext.lower() for ext in allowed_extensions]

def safe_filename(filename: str) -> str:
    """Generate a safe filename by removing/replacing problematic characters"""
    import re
    
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

def get_unique_filename(directory: str, filename: str) -> str:
    """Generate a unique filename in the given directory"""
    if not os.path.exists(os.path.join(directory, filename)):
        return filename
    
    name, ext = os.path.splitext(filename)
    counter = 1
    
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1

def copy_file_with_progress(src: str, dst: str, chunk_size: int = 8192):
    """Copy file with progress tracking"""
    try:
        total_size = os.path.getsize(src)
        copied = 0
        
        with open(src, 'rb') as src_file:
            with open(dst, 'wb') as dst_file:
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    dst_file.write(chunk)
                    copied += len(chunk)
                    
                    # Log progress every 10MB
                    if copied % (10 * 1024 * 1024) == 0:
                        progress = (copied / total_size) * 100
                        logger.info(f"Copy progress: {progress:.1f}%")
        
        logger.info(f"Successfully copied {src} to {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying file {src} to {dst}: {e}")
        return False

def get_directory_size(directory: str) -> int:
    """Get total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.error(f"Error calculating directory size: {e}")
    
    return total_size

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