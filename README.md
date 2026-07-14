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
*   `dataset/`: Directory to store the raw fashion images.
*   `evaluation/`: Scripts to evaluate the system on predefined queries.
*   `indexer/`: Modules to load images, generate captions, extract metadata, create embeddings, and populate the database.
*   `models/`: Directory for any locally saved models (if applicable).
*   `notebooks/`: Directory for Jupyter notebooks.
*   `retriever/`: Modules to parse queries, search the database, and perform hybrid reranking.
*   `utils/`: Helper utilities (logging, config parsing).
*   `vectordb/`: Persistent storage for ChromaDB.

## Installation

1.  Ensure you have Python 3.8+ installed.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

This system is designed to work with the Fashionpedia dataset or any directory of images.

For testing purposes, place a subset of images (e.g., 500-1000) inside the `dataset/images/` directory.

## Usage

### 1. Indexing the Dataset

Run the indexer to process images, generate captions, extract metadata, and store embeddings in ChromaDB:
```bash
python indexer/index_dataset.py
```

### 2. Searching

You can test individual queries using the main retriever script:
```bash
python retriever/main.py "A person in a bright yellow raincoat"
```

### 3. Evaluation

Run the evaluation script to test the system against the predefined assignment queries:
```bash
python evaluation/eval.py
```

### 4. Visualization

To visually verify that the retrieved images match the query, use the visualization script. This will open a window showing the query and the top 3 image results side-by-side:
```bash
python evaluation/visualize.py "A person in a bright yellow raincoat"
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
- Add a web interface (e.g., using Streamlit or Gradio) for interactive testing.
