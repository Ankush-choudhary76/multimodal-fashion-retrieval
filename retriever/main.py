import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config
from utils.logger import get_logger
from indexer.clip_encoder import CLIPEncoder
from indexer.vector_store import VectorStore
from retriever.query_parser import QueryParser
from retriever.query_encoder import QueryEncoder
from retriever.search import Searcher
from retriever.reranker import Reranker
from typing import List, Dict, Any

logger = get_logger(__name__)

class FashionRetriever:
    def __init__(self):
        self.config = load_config()
        self.logger = logger
        
        self.logger.info("Initializing Fashion Retriever...")
        
        self.clip_enc = CLIPEncoder(
            model_name=self.config['models']['clip_model'],
            pretrained=self.config['models']['clip_pretrained'],
            device=self.config['models']['device']
        )
        self.vector_store = VectorStore(db_path=self.config['project']['vector_db_path'])
        
        self.query_parser = QueryParser()
        self.query_encoder = QueryEncoder(self.clip_enc)
        self.searcher = Searcher(self.vector_store)
        self.reranker = Reranker(self.config['retrieval']['weights'])
        
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Executes the full retrieval pipeline.
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
            
        self.logger.info(f"Processing query: '{query}'")
        
        # 1. Parse Query
        parsed_query = self.query_parser.parse(query)
        self.logger.info(f"Extracted Attributes: {parsed_query}")
        
        # 2. Encode Query
        query_embedding = self.query_encoder.encode(query)
        if not query_embedding:
            self.logger.error("Failed to encode query.")
            return []
            
        # 3. Vector Search (Fetch more candidates than top_k for reranking)
        fetch_k = max(100, top_k * 5)
        raw_results = self.searcher.search(query_embedding, top_k=fetch_k)
        
        # 4. Hybrid Reranking
        reranked_results = self.reranker.rerank(raw_results, parsed_query)
        
        # 5. Return Top K
        return reranked_results[:top_k]

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py 'your search query'")
        return
        
    query = " ".join(sys.argv[1:])
    retriever = FashionRetriever()
    results = retriever.retrieve(query)
    
    print(f"\n--- Top {len(results)} Results ---")
    for i, res in enumerate(results):
        print(f"\nRank {i+1}:")
        print(f"Image URI: {res['uri']}")
        print(f"Final Score: {res['score']:.4f} (CLIP Sim: {res['clip_sim']:.4f})")
        print(f"Metadata: {res['metadata']}")

if __name__ == "__main__":
    main()
