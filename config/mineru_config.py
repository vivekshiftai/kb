import json
import os
from config.settings import get_settings

def create_mineru_config(output_path: str = None) -> str:
    """Create MinerU configuration file"""
    settings = get_settings()
    
    config = {
        "weights_path": settings.MODELS_DIR,
        "device-mode": settings.DEVICE_MODE,
        "models_dir": settings.MODELS_DIR,
        "models-dir": settings.MODELS_DIR,
        "virtual-vram-size": settings.VIRTUAL_VRAM_SIZE,
        "method": "auto",
        "backend": "pipeline",
        "formula-enable": True,
        "table-enable": True,
        "start-page": 0,
        "end-page": None
    }
    
    if output_path is None:
        output_path = os.path.expanduser("~/magic-pdf.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return output_path

def get_mineru_config_path() -> str:
    """Get the path to MinerU configuration file"""
    return os.path.expanduser("~/magic-pdf.json")