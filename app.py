import streamlit as st
import os
import sys
from PIL import Image

# Ensure the retriever can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever.main import FashionRetriever

# Set page config for a wider, cleaner layout
st.set_page_config(page_title="Glance ML | Fashion Retrieval", page_icon="👗", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 15px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] {
        background-color: #f4f6f9;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_retriever():
    """Loads the Fashion Retriever model only once."""
    return FashionRetriever()

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1200px-GitHub_Invertocat_Logo.svg.png", width=50)
        st.markdown("### Glance ML Internship")
        st.markdown("**Multimodal Fashion Retrieval**")
        st.markdown("---")
        
        st.markdown("### ⚙️ Engine Settings")
        top_k = st.slider("Results to return (Top-K):", 1, 10, 3)
        
        st.markdown("#### Hybrid Scoring Weights")
        st.caption("Tune semantic vs. attribute matching.")
        w_clip = st.slider("CLIP Semantic Sim", 0.0, 1.0, 0.5)
        w_cloth = st.slider("Clothing Match", 0.0, 1.0, 0.2)
        w_color = st.slider("Color Match", 0.0, 1.0, 0.2)
        w_scene = st.slider("Scene Match", 0.0, 1.0, 0.1)
        
        st.markdown("---")

    # --- MAIN CONTENT ---
    st.markdown('<p class="main-header">Fashion Context Retrieval Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic search powered by OpenCLIP, BLIP, and ChromaDB.</p>', unsafe_allow_html=True)

    # Load Model
    with st.spinner("Initializing Models and Vector Database (ChromaDB)..."):
        retriever = load_retriever()
        
    # Update retriever weights dynamically from sidebar
    retriever.reranker.weights = {
        "clip": w_clip,
        "clothing": w_cloth,
        "color": w_color,
        "scene": w_scene
    }

    # Search Bar
    query = st.text_input("🔍 Search Database:", placeholder="Describe an outfit, color, or scene...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("Search Images", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner(f"Searching database for '{query}'..."):
            results = retriever.retrieve(query, top_k=top_k)
            
            if not results:
                st.warning("No results found. Try a different query.")
            else:
                st.markdown(f"### ✨ Top {len(results)} Results for: *{query}*")
                
                # Display results in columns
                cols = st.columns(len(results))
                for idx, (col, res) in enumerate(zip(cols, results)):
                    with col:
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        try:
                            img = Image.open(res['uri'])
                            st.image(img, use_container_width=True)
                            
                            st.markdown(f"**Rank {idx + 1}** | Score: `{res['score']:.3f}`")
                            
                            meta = res['metadata']
                            st.markdown(f"**👕 Cloth:** {meta.get('clothing', '-')}")
                            st.markdown(f"**🎨 Color:** {meta.get('colors', '-')}")
                            st.markdown(f"**📍 Scene:** {meta.get('scenes', '-')}")
                            
                            with st.expander("Show BLIP Caption"):
                                st.caption(meta.get('caption', 'N/A'))
                        except Exception as e:
                            st.error(f"Error loading image: {res['uri']}")
                        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
