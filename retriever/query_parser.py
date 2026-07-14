from indexer.metadata_extractor import MetadataExtractor
from typing import Dict, List

class QueryParser:
    def __init__(self):
        # We reuse the vocabulary from the metadata extractor
        self.extractor = MetadataExtractor()

    def parse(self, query: str) -> Dict[str, List[str]]:
        """
        Parses the natural language query to extract relevant structured attributes.
        """
        return self.extractor.extract(query)
