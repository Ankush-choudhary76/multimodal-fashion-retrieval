import os
from PIL import Image
from typing import List, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class ImageLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def get_image_paths(self, max_images: int = 1000) -> List[str]:
        """
        Retrieves a list of image paths from the dataset directory.
        """
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset path does not exist: {self.dataset_path}")
            return []

        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_paths = []
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
                    if len(image_paths) >= max_images:
                        break
            if len(image_paths) >= max_images:
                break
                
        logger.info(f"Found {len(image_paths)} images in {self.dataset_path}")
        return image_paths

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from the given path.
        """
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
