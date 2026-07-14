import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.main import FashionRetriever

def run_evaluation():
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    
    print("Loading Retriever System...")
    retriever = FashionRetriever()
    
    print("\n" + "="*50)
    print("FASHION RETRIEVAL EVALUATION")
    print("="*50)
    
    for query in queries:
        print(f"\n[Query]: {query}")
        print("-" * 50)
        
        results = retriever.retrieve(query, top_k=3)
        
        if not results:
            print("No results found.")
            continue
            
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['uri']}")
            print(f"     Score: {res['score']:.4f}")
            print(f"     Metadata:")
            print(f"       - Clothing: {res['metadata'].get('clothing', '')}")
            print(f"       - Colors: {res['metadata'].get('colors', '')}")
            print(f"       - Scenes: {res['metadata'].get('scenes', '')}")
            print(f"       - BLIP Caption: {res['metadata'].get('caption', '')}")
        print("="*50)

if __name__ == "__main__":
    run_evaluation()
