# Glance ML Internship Assignment: Multimodal Fashion & Context Retrieval

## 1. Approaches: Possible Ways to Solve the Problem

When tackling multimodal fashion retrieval, there are several possible architectural approaches, each with its own tradeoffs:

**Approach A: Vanilla CLIP (Zero-Shot Vector Search)**
*   **How it works:** Encode both images and text queries using pre-trained CLIP models and perform a K-Nearest Neighbors (KNN) search using cosine similarity.
*   **Pros:** Extremely easy to implement; requires no training; highly scalable.
*   **Cons:** Struggles with compositionality (e.g., "red shirt and blue pants" vs "blue shirt and red pants"). It captures the overall "vibe" but often ignores fine-grained fashion attributes.

**Approach B: Fine-Tuning CLIP (Domain Adaptation)**
*   **How it works:** Fine-tune the CLIP image and text encoders on a specific fashion dataset (like Fashionpedia) using a contrastive loss objective.
*   **Pros:** Significantly improves performance on domain-specific fashion terminology.
*   **Cons:** Requires large amounts of high-quality paired image-text training data and significant compute resources.

**Approach C: Hybrid Retrieval (VLM Metadata + Vector Search) [CHOSEN APPROACH]**
*   **How it works:** Use a Vision-Language Model (VLM) like BLIP to generate dense captions for images. Extract explicit structured metadata (colors, clothing types, scenes) from these captions. During retrieval, combine CLIP's vector similarity score with an explicit attribute-matching score (Jaccard similarity) from the metadata.
*   **Pros:** Directly addresses CLIP's compositionality weakness by forcing the system to respect explicit keywords (like specific colors and garments) while maintaining zero-shot semantic understanding. Does not require fine-tuning.
*   **Cons:** Slower indexing time due to the BLIP captioning step.

---

## 2. Chosen Approach: Hybrid Metadata & Semantic Retrieval

We chose **Approach C (Hybrid Retrieval)** because it perfectly addresses the assignment's hint regarding vanilla CLIP's struggle with compositionality, prioritizing ML logic over brute-force engineering.

**Architecture:**
1.  **Indexer:** Images are passed through `Salesforce/blip-image-captioning-base` to generate natural language descriptions. A metadata extractor parses these captions for explicit attributes (clothing type, color, scene). The image is also embedded using OpenCLIP (`ViT-B-32`). Both the embedding and the extracted structured metadata are stored in **ChromaDB**.
2.  **Retriever:** The user's query is parsed to extract target attributes and encoded using OpenCLIP. ChromaDB fetches the Top-K (e.g., 100) semantically similar images.
3.  **Hybrid Reranking:** A custom reranker evaluates the Top-K images by calculating a weighted final score:
    `Final Score = (w1 * CLIP_Similarity) + (w2 * Color_Match) + (w3 * Clothing_Match) + (w4 * Scene_Match)`

**How it handles fashion queries:**
If a user searches for *"A red tie and a white shirt in a formal setting"*, vanilla CLIP might return an image of a red shirt and a white tie. Our hybrid approach avoids this by explicitly extracting "red", "white", "tie", and "shirt" from the query and heavily penalizing candidates in the reranking phase that do not contain these explicit attributes in their BLIP-generated metadata.

---

## 3. Codebase (GitHub) Link
**Repository:** [https://github.com/Ankush-choudhary76/multimodal-fashion-retrieval](https://github.com/Ankush-choudhary76/multimodal-fashion-retrieval)
*(Note: Code is modularly separated into `indexer` and `retriever` pipelines, and the test dataset/ChromaDB are included for plug-and-play testing).*

---

## 4. Approaches for Future Work

**A. Extending the solution for adding locations (cities, places) and weather:**
*   **Weather Extraction:** We can integrate an LLM or a specialized Named Entity Recognition (NER) model to parse implied weather conditions from queries (e.g., "sunny day", "winter outfit") and match them against visual cues extracted by a more powerful VLM (like LLaVA) during indexing (e.g., detecting snow, bright sunlight, umbrellas).
*   **Location Integration:** If images have EXIF data (GPS coordinates), we can implement a Geo-Spatial filter in ChromaDB. The query parser can extract city names using an external Geocoding API and filter the vector database to only search within a specific radius of that location before applying vector similarity.

**B. How to improve precision:**
*   **Cross-Encoder Reranking:** Replace the linear weighted Jaccard similarity reranker with a pre-trained Vision-Language Cross-Encoder. While bi-encoders (like CLIP) are fast for initial retrieval, cross-encoders process the image and query simultaneously with self-attention, yielding vastly higher precision for complex compositional queries.
*   **Dense Structured Captioning:** Instead of using a base BLIP model, use a state-of-the-art multimodal LLM (like GPT-4o-mini or LLaVA) during the indexing phase to generate a highly structured JSON output for every image (e.g., `{"garments": [{"type": "shirt", "color": "blue", "material": "denim"}], "vibe": "casual"}`). This allows for exact SQL-like filtering alongside vector search.
