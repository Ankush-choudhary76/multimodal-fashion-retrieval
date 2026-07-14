import re
from typing import Dict, List, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class MetadataExtractor:
    def __init__(self):
        # Define simple vocabularies for extracting attributes
        self.clothing_types = [
            "shirt", "t-shirt", "dress", "pants", "jeans", "shorts", "skirt", 
            "jacket", "coat", "sweater", "hoodie", "suit", "blazer", "raincoat", 
            "tie", "scarf", "hat", "shoes", "sneakers", "boots", "heels", "glasses", "sunglasses"
        ]
        
        self.colors = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
            "pink", "purple", "orange", "brown", "beige", "navy", "maroon", "cyan", "bright"
        ]
        
        self.scenes = [
            "office", "park", "street", "home", "beach", "city", "indoor", "outdoor", 
            "room", "building", "nature", "setting", "background"
        ]

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts clothing types, colors, and scenes from a given text (caption or query).
        """
        text = text.lower()
        # Simple word tokenization, stripping punctuation
        words = re.findall(r'\b\w+\b', text)
        
        extracted = {
            "clothing": [],
            "colors": [],
            "scenes": []
        }
        
        for word in words:
            if word in self.clothing_types and word not in extracted["clothing"]:
                extracted["clothing"].append(word)
            if word in self.colors and word not in extracted["colors"]:
                extracted["colors"].append(word)
            if word in self.scenes and word not in extracted["scenes"]:
                extracted["scenes"].append(word)
                
        return extracted
        
    def to_metadata_dict(self, extracted: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Converts the extracted lists into a flat dictionary suitable for ChromaDB metadata.
        """
        return {
            "clothing": ",".join(extracted["clothing"]) if extracted["clothing"] else "",
            "colors": ",".join(extracted["colors"]) if extracted["colors"] else "",
            "scenes": ",".join(extracted["scenes"]) if extracted["scenes"] else ""
        }
