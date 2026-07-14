# Multimodal Fashion Retrieval System

A production-ready multimodal fashion image retrieval system built with PyTorch, OpenCLIP, BLIP, and ChromaDB.

This system retrieves fashion images based on natural language queries using a hybrid ranking approach, going beyond vanilla CLIP by incorporating explicit attribute matching (clothing type, color, scene) extracted from generated image captions.

## Project Architecture

1.  **Image Preprocessing**: Images are loaded and preprocessed.
2.  **BLIP Image Captioning**: `Salesforce/blip-image-captioning-base` generates descriptive captions for the images.
3.  **Metadata Extraction**: Structured attributes (clothing, colors, scenes) are extracted from the BLIP captions using predefined vocabularies.
4.  **OpenCLIP Embedding**: `ViT-B-32` generates vector embeddings for the images.
5.  **Vector Database**: ChromaDB stores the image embeddings and the structured metadata.
6.  **Retrieval & Hybrid Scoring**:
    *   The user's text query is parsed for attributes and encoded via OpenCLIP.
    *   ChromaDB performs an initial vector search to retrieve the top candidates.
    *   A hybrid scoring function reranks candidates based on CLIP similarity and explicit attribute overlap (Jaccard similarity).

## Folder Structure

*   `configs/`: Configuration files (`config.yaml`).
*   `dataset/`: Directory storing the raw fashion images (1,000 images included for testing).
*   `evaluation/`: Scripts to evaluate the system on predefined queries.
*   `indexer/`: Modules to load images, generate captions, extract metadata, create embeddings, and populate the database.
*   `retriever/`: Modules to parse queries, search the database, and perform hybrid reranking.
*   `utils/`: Helper utilities (logging, config parsing).
*   `vectordb/`: Persistent storage for ChromaDB (Pre-indexed database included).

## Installation

1.  Ensure you have Python 3.8+ installed.
2.  Clone the repository and install the required dependencies:
    ```bash
    git clone https://github.com/Ankush-choudhary76/multimodal-fashion-retrieval.git
    cd multimodal-fashion-retrieval
    pip install -r requirements.txt
    ```

## Usage (Plug-and-Play)

We have included a pre-indexed vector database (`vectordb/`) and 1,000 test images (`dataset/images/`) directly in the repository. **You do not need to run the indexer yourself.** You can immediately test the retrieval!

### 1. Interactive Web UI (Recommended)

To launch the beautiful, interactive Streamlit web application:
```bash
streamlit run app.py
```
This UI allows you to test queries instantly and adjust the hybrid scoring weights via sliders!

### 2. Command-Line Evaluation

Run the evaluation script to test the system against the predefined assignment queries:
```bash
python evaluation/eval.py
```

### 3. Command-Line Visualization

To visually verify that the retrieved images match a specific custom query:
```bash
python evaluation/visualize.py "A person in a bright yellow raincoat"
```

### 4. Indexing (Optional)

If you wish to add new images or rebuild the database from scratch, place your images in `dataset/images/` and run:
```bash
python indexer/index_dataset.py
```

## Configuration

You can customize the system behavior by editing `configs/config.yaml`:
*   `dataset_path`: Path to the image directory.
*   `vector_db_path`: Path to store the ChromaDB data.
*   `max_index_images`: Limit the number of images to index.
*   `models`: Change the BLIP or OpenCLIP models.
*   `weights`: Adjust the importance of CLIP vs. explicit attributes in the hybrid scoring.

## Future Improvements
- Replace rule-based attribute extraction with a lightweight Named Entity Recognition (NER) model or an LLM for more robust metadata extraction.
- Implement a more complex reranking model (e.g., Cross-Encoder) instead of a linear weighted sum.
