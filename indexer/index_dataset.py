import os
import sys
from tqdm import tqdm

# Add the parent directory to sys.path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config
from utils.logger import get_logger
from indexer.image_loader import ImageLoader
from indexer.caption_generator import CaptionGenerator
from indexer.metadata_extractor import MetadataExtractor
from indexer.clip_encoder import CLIPEncoder
from indexer.vector_store import VectorStore

logger = get_logger(__name__)

def main():
    config = load_config()
    
    dataset_path = config['project']['dataset_path']
    max_images = config['project']['max_index_images']
    
    # Initialize components
    logger.info("Initializing indexer components...")
    image_loader = ImageLoader(dataset_path)
    
    image_paths = image_loader.get_image_paths(max_images)
    if not image_paths:
        logger.warning("No images found to index. Exiting.")
        return
        
    caption_gen = CaptionGenerator(
        model_id=config['models']['blip_model'], 
        device=config['models']['device']
    )
    metadata_ext = MetadataExtractor()
    clip_enc = CLIPEncoder(
        model_name=config['models']['clip_model'],
        pretrained=config['models']['clip_pretrained'],
        device=config['models']['device']
    )
    vector_store = VectorStore(db_path=config['project']['vector_db_path'])
    
    logger.info(f"Starting indexing for {len(image_paths)} images...")
    
    for idx, path in enumerate(tqdm(image_paths, desc="Indexing Images")):
        # Load image
        img = image_loader.load_image(path)
        if img is None:
            continue
            
        # Generate Caption
        caption = caption_gen.generate_caption(img)
        
        # Extract Metadata
        extracted = metadata_ext.extract(caption)
        metadata = metadata_ext.to_metadata_dict(extracted)
        # Add the raw caption to metadata as well
        metadata["caption"] = caption
        
        # Generate Embedding
        embedding = clip_enc.encode_image(img)
        
        if not embedding:
            logger.warning(f"Failed to generate embedding for {path}")
            continue
            
        # Store in Vector DB
        image_id = f"img_{idx}"
        vector_store.add_item(image_id, embedding, metadata, uri=path)
        
    logger.info("Indexing completed successfully!")

if __name__ == "__main__":
    main()
