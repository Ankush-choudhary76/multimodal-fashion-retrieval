import streamlit as st
import os
import sys
from PIL import Image

# Ensure the retriever can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever.main import FashionRetriever

# Set page config
st.set_page_config(page_title="Fashion Retrieval Engine", layout="wide")

@st.cache_resource
def load_retriever():
    """Loads the Fashion Retriever model only once."""
    return FashionRetriever()

def main():
    st.title("👗 Multimodal Fashion & Context Retrieval")
    st.markdown("Search for fashion images using natural language queries. The system uses a **Hybrid Retrieval** approach combining OpenCLIP semantics with explicit BLIP-extracted metadata.")

    # Load Model
    with st.spinner("Loading ML Models and Vector Database..."):
        retriever = load_retriever()

    # Search Bar
    query = st.text_input("Enter your fashion query:", placeholder="e.g., A person in a bright yellow raincoat.")
    
    # Sliders for Hybrid Weights (Optional interactivity for bonus points)
    with st.expander("⚙️ Adjust Hybrid Scoring Weights"):
        st.markdown("Tune how much weight is given to the neural vector search vs explicit keyword matching.")
        col1, col2, col3, col4 = st.columns(4)
        w_clip = col1.slider("CLIP Semantic Sim", 0.0, 1.0, 0.5)
        w_cloth = col2.slider("Clothing Match", 0.0, 1.0, 0.2)
        w_color = col3.slider("Color Match", 0.0, 1.0, 0.2)
        w_scene = col4.slider("Scene Match", 0.0, 1.0, 0.1)
        
        # Update retriever weights dynamically
        retriever.reranker.weights = {
            "clip": w_clip,
            "clothing": w_cloth,
            "color": w_color,
            "scene": w_scene
        }

    top_k = st.slider("Number of results to return (Top-K):", 1, 10, 3)

    if st.button("Search 🔍") and query:
        with st.spinner(f"Searching database for '{query}'..."):
            results = retriever.retrieve(query, top_k=top_k)
            
            if not results:
                st.warning("No results found.")
            else:
                st.success(f"Top {len(results)} results retrieved!")
                
                # Display results in columns
                cols = st.columns(len(results))
                for idx, (col, res) in enumerate(zip(cols, results)):
                    with col:
                        try:
                            img = Image.open(res['uri'])
                            st.image(img, use_container_width=True)
                            
                            st.markdown(f"**Rank {idx + 1}**")
                            st.metric(label="Hybrid Score", value=f"{res['score']:.3f}")
                            
                            meta = res['metadata']
                            st.caption(f"**Extracted Cloth:** {meta.get('clothing', 'N/A')}")
                            st.caption(f"**Extracted Color:** {meta.get('colors', 'N/A')}")
                            with st.expander("View BLIP Caption"):
                                st.write(meta.get('caption', 'N/A'))
                        except Exception as e:
                            st.error(f"Error loading image: {res['uri']}")

if __name__ == "__main__":
    main()
