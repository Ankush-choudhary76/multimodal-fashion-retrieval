import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.logger import get_logger

logger = get_logger(__name__)

class CaptionGenerator:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "cpu"):
        self.device = device
        logger.info(f"Loading BLIP model: {model_id} on {self.device}")
        try:
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise e

    def generate_caption(self, image: Image.Image) -> str:
        """
        Generates a caption for the given image.
        """
        if image is None:
            return ""
            
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return ""
