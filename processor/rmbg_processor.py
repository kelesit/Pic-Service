from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import cv2
import numpy as np

from processor.image_processor import ImageProcessor


logger = logging.getLogger(__name__)


class RMBGProcessor(ImageProcessor):
    """背景移除处理器"""

    def __init__(self):
        super().__init__()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载背景移除模型"""
        try:
            pass
        except Exception as e:
            logger.error(f"背景移除模型加载失败: {str(e)}")
            raise

    def remove_background(
            self,
            image_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
    ):
        try:
            self.validate_image(image_path)
            image_path = Path(image_path)
            
            # 如果未指定输出路径，在原文件旁创建
            if output_path is None:
                output_path = image_path.parent / f"{image_path.stem}_nobg{image_path.suffix}"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # 模拟处理流程
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgba = cv2.cvtColor(gray, cv2.COLOR_BGRA2RGBA)
        
            # 保存处理后的图片
            success = cv2.imwrite(str(output_path), rgba)
            if not success:
                raise IOError(f"Failed to save image to {output_path}")
                
            return output_path
        
        except Exception as e:
            logger.error(f"Background removal failed: {str(e)}")
            raise

