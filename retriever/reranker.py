from typing import List, Dict, Any

class Reranker:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def _calculate_attribute_similarity(self, query_attrs: List[str], img_attrs_str: str) -> float:
        """
        Calculates a simple Jaccard-like similarity for attributes.
        """
        if not query_attrs:
            return 1.0 # If no attributes specified in query, ignore this score component
            
        img_attrs = [attr.strip() for attr in img_attrs_str.split(',')] if img_attrs_str else []
        
        if not img_attrs:
            return 0.0
            
        intersection = len(set(query_attrs).intersection(set(img_attrs)))
        # Normalize by query length so finding all requested attributes gives score 1.0
        return intersection / len(query_attrs)

    def rerank(self, raw_results: Dict[str, Any], parsed_query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Reranks the raw results based on the hybrid scoring formula.
        raw_results format: {"ids": [[id1, id2]], "distances": [[d1, d2]], "metadatas": [[{m1}, {m2}]]}
        """
        if not raw_results or not raw_results["ids"] or not raw_results["ids"][0]:
            return []

        reranked_results = []
        ids = raw_results["ids"][0]
        distances = raw_results["distances"][0] # ChromaDB returns distance, smaller is better (usually 1 - cosine_sim)
        metadatas = raw_results["metadatas"][0]

        for i in range(len(ids)):
            # Convert distance to similarity (assuming cosine distance from ChromaDB)
            clip_sim = max(0, 1.0 - distances[i]) 
            
            meta = metadatas[i]
            
            cloth_sim = self._calculate_attribute_similarity(parsed_query["clothing"], meta.get("clothing", ""))
            color_sim = self._calculate_attribute_similarity(parsed_query["colors"], meta.get("colors", ""))
            scene_sim = self._calculate_attribute_similarity(parsed_query["scenes"], meta.get("scenes", ""))
            
            final_score = (
                self.weights.get("clip", 0.5) * clip_sim +
                self.weights.get("clothing", 0.2) * cloth_sim +
                self.weights.get("color", 0.2) * color_sim +
                self.weights.get("scene", 0.1) * scene_sim
            )
            
            reranked_results.append({
                "id": ids[i],
                "uri": meta.get("uri", ""),
                "score": final_score,
                "clip_sim": clip_sim,
                "metadata": meta
            })
            
        # Sort by final score descending
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results
