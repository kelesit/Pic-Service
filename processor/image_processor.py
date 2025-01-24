from pathlib import Path
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理核心类"""

    def __init__(self):
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.webp'}
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """验证图像文件"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if image_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {str(e)}")
            raise

