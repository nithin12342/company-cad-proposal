"""Model downloader and management utilities.

Handles automatic downloading of production models with hardware detection.
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional
import logging

import torch

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Manages automatic download of production ML models."""

    MODELS = {
        "sam_vit_h_4b8939.pth": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size": 638803384,  # ~609MB
            "checksum": "89400e0a8b0e6c7b8e7c8f9a0b1c2d3e4f5a6b7c8",  # partial
        },
        "doctr": {
            "package": "python-doctr",
            "url": None,
        }
    }

    def __init__(self, model_dir: str = "./models"):
        """Initialize downloader.
        
        Args:
            model_dir: Directory to store downloaded models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_device() -> torch.device:
        """Detect and return optimal hardware device.
        
        Returns:
            torch.device: cuda, mps, or cpu
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.warning("Using CPU - performance will be limited")
        return device

    def ensure_model(self, model_name: str) -> Path:
        """Ensure model file exists, download if missing.
        
        Args:
            model_name: Name of model file (e.g., sam_vit_h_4b8939.pth)
            
        Returns:
            Path to model file
            
        Raises:
            RuntimeError: If download fails or model unavailable
        """
        if model_name not in self.MODELS:
            raise RuntimeError(f"Unknown model: {model_name}")

        model_path = self.model_dir / model_name

        if model_path.exists():
            logger.info(f"Model found: {model_path}")
            return model_path

        # Download model
        model_info = self.MODELS[model_name]
        if not model_info["url"]:
            raise RuntimeError(
                f"Model '{model_name}' requires manual installation. "
                f"Please install via: pip install {model_info['package']}"
            )

        logger.info(f"Downloading {model_name}...")
        logger.info(f"URL: {model_info['url']}")
        
        try:
            self._download_file(model_info["url"], model_path)
            logger.info(f"Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {model_name}: {e}\n"
                f"Please download manually from: {model_info['url']}\n"
                f"Then place at: {model_path}"
            )

    def _download_file(self, url: str, dest_path: Path):
        """Download file with progress indicator.
        
        Args:
            url: Source URL
            dest_path: Destination path
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownloading: {percent}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, dest_path, reporthook)
            print()  # Newline after progress
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()
            raise RuntimeError(f"Download failed: {e}")

    def check_doctr(self) -> bool:
        """Check if python-doctr is installed.
        
        Returns:
            True if available
            
        Raises:
            RuntimeError: If not available
        """
        try:
            import doctr  # noqa: F401
            from doctr import models  # noqa: F401
            logger.info("python-doctr is available")
            return True
        except ImportError:
            raise RuntimeError(
                "python-doctr is not installed.\n"
                "Please install with: pip install python-doctr@git+https://github.com/mindee/doctr.git"
            )

    def check_openv(self) -> bool:
        """Check if OpenCV is installed.
        
        Returns:
            True if available
            
        Raises:
            RuntimeError: If not available
        """
        try:
            import cv2
            logger.info(f"OpenCV {cv2.__version__} is available")
            return True
        except ImportError:
            raise RuntimeError(
                "OpenCV is not installed.\n"
                "Please install with: pip install opencv-python-headless"
            )


def setup_production() -> dict:
    """Setup production environment with all required dependencies.
    
    Returns:
        Dict with device and model paths
        
    Raises:
        RuntimeError: If any required dependency is missing
    """
    downloader = ModelDownloader()
    
    # Check dependencies
    downloader.check_openv()
    downloader.check_doctr()
    
    # Download models
    sam_path = downloader.ensure_model("sam_vit_h_4b8939.pth")
    
    # Get device
    device = downloader.get_device()
    
    logger.info("Production environment ready")
    
    return {
        "device": device,
        "sam_path": sam_path,
        "model_dir": downloader.model_dir,
    }