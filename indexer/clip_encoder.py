import torch
import open_clip
from PIL import Image
from typing import List, Union
from utils.logger import get_logger

logger = get_logger(__name__)

class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        self.device = device
        logger.info(f"Loading OpenCLIP model: {model_name} ({pretrained}) on {self.device}")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise e

    def encode_image(self, image: Image.Image) -> List[float]:
        """
        Encodes a single image into a vector embedding.
        """
        if image is None:
            return []
            
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return []

    def encode_text(self, text: str) -> List[float]:
        """
        Encodes a text string into a vector embedding.
        """
        try:
            text_input = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return []
