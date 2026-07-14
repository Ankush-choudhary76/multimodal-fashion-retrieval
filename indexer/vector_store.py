import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, db_path: str = "vectordb/chroma_data", collection_name: str = "fashion_images"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_item(self, image_id: str, embedding: List[float], metadata: Dict[str, str], uri: str):
        """
        Adds a single item to the vector store.
        """
        # Ensure metadata contains only strings/ints/floats/bools for ChromaDB
        sanitized_metadata = {k: str(v) for k, v in metadata.items()}
        sanitized_metadata["uri"] = uri
        
        try:
            self.collection.upsert(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[sanitized_metadata],
            )
        except Exception as e:
            logger.error(f"Failed to add item {image_id} to ChromaDB: {e}")

    def search(self, query_embedding: List[float], top_k: int = 100) -> Dict[str, Any]:
        """
        Searches for the most similar items based on vector similarity.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return {"ids": [], "distances": [], "metadatas": []}
