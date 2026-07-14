import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retriever.main import FashionRetriever

def visualize_query(query: str, top_k: int = 3):
    print(f"Loading retriever and searching for: '{query}'...")
    retriever = FashionRetriever()
    results = retriever.retrieve(query, top_k=top_k)
    
    if not results:
        print("No results found.")
        return

    # Plot the results
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    fig.suptitle(f"Query: {query}", fontsize=16)

    # Handle case where top_k is 1 (axes is not an array)
    if top_k == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        img_path = res['uri']
        score = res['score']
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            
            # Format the title to show score and extracted attributes
            title = f"Score: {score:.3f}\n"
            meta = res['metadata']
            if meta.get('clothing'): title += f"Cloth: {meta['clothing']}\n"
            if meta.get('colors'): title += f"Color: {meta['colors']}"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.set_title("Image Load Error")
            ax.axis('off')
            print(f"Failed to load image {img_path}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation/visualize.py 'your search query'")
    else:
        user_query = " ".join(sys.argv[1:])
        visualize_query(user_query)
