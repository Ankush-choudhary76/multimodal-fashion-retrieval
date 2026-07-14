from indexer.vector_store import VectorStore
from typing import List, Dict, Any

class Searcher:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query_embedding: List[float], top_k: int = 100) -> Dict[str, Any]:
        """
        Retrieves the top_k most similar images from the vector store based on the query embedding.
        Returns a dictionary containing 'ids', 'distances', and 'metadatas'.
        """
        return self.vector_store.search(query_embedding, top_k=top_k)
