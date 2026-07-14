from indexer.clip_encoder import CLIPEncoder
from typing import List

class QueryEncoder:
    def __init__(self, clip_encoder: CLIPEncoder):
        """
        Takes an already initialized CLIPEncoder to avoid reloading the model.
        """
        self.encoder = clip_encoder

    def encode(self, query: str) -> List[float]:
        """
        Encodes the query text into a vector embedding.
        """
        return self.encoder.encode_text(query)
