from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image

from processor.image_processor import ImageProcessor
from image_proc_utils import refine_foreground

logger = logging.getLogger(__name__)


class RMBGProcessor(ImageProcessor):
    """背景移除处理器"""
    MODEL_DIR = '/root/autodl-tmp/smart_inpaint_models/RMBG-2.0'
    INPUT_IMAGE_SIZE = (1024, 1024)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        self.model = None
        self.transform_image = transforms.Compose([
            transforms.Resize(self.INPUT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(self.NORMALIZE_MEAN, self.NORMALIZE_STD)
        ])
        self._load_model()


    def _load_model(self):
        """加载背景移除模型"""
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(self.MODEL_DIR, trust_remote_code=True)
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.model.to('cuda')
            self.model.eval()
        except Exception as e:
            logger.error(f"背景移除模型加载失败: {str(e)}")
            raise


    def predict(self, input_images, original_size):
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        return pred_pil

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
            # img = cv2.imread(str(image_path))
            img = Image.open(image_path).convert("RGB")
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            original_size = img.size
            # process img
            input_images = self.transform_image(img).unsqueeze(0).to('cuda')
            # predict
            pred_pil = self.predict(input_images, original_size)
            # post process
            image_masked = refine_foreground(img, pred_pil)
            image_masked.putalpha(pred_pil.resize(img.size))

            # 创建白色背景
            white_background = Image.new("RGBA", original_size, (255, 255, 255, 255))
            # 将透明图像与白底合成
            composite = Image.alpha_composite(white_background, image_masked)
            final_img = composite.convert("RGB")
            
            # 转换为numpy数组并调整颜色通道
            final_array = np.array(final_img)
            final_array = cv2.cvtColor(final_array, cv2.COLOR_RGB2BGR)
            # 保存处理后的图片
            success = cv2.imwrite(str(output_path), final_array)
            if not success:
                raise IOError(f"Failed to save image to {output_path}")
                
            return output_path
        
        except Exception as e:
            logger.error(f"Background removal failed: {str(e)}")
            raise

