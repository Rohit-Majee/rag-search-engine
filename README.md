### `README.md`

```markdown
# 🔍 Custom RAG Search Engine From Scratch

A full-featured, from-scratch Retrieval-Augmented Generation (RAG) search engine built in Python. 

Instead of just wrapping high-level libraries, this project implements the mathematical and architectural fundamentals of modern search engines from the ground up—starting with raw text processing and TF-IDF, moving through BM25 and Semantic Search, and culminating in advanced Hybrid ranking, RAG, and Multimodal (Image + Text) search.

## ✨ Core Features
* **Keyword Search (BM25):** Custom implementation of Term Frequency (TF), Inverse Document Frequency (IDF), and the industry-standard BM25 algorithm for exact-match retrieval.
* **Semantic Vector Search:** Uses `sentence-transformers` and local HuggingFace embeddings to search documents by meaning and context rather than exact keywords.
* **Advanced Text Chunking:** Includes fixed-size and semantic chunking with overlapping sliding windows to preserve context in long documents.
* **Hybrid Search (RRF & Weighted):** Combines keyword and semantic search results using Reciprocal Rank Fusion (RRF) and Weighted Averages with customizable alpha coefficients.
* **LLM Query Enhancements:** Leverages the Gemini API to automatically fix spelling, rewrite, or expand queries prior to vector retrieval.
* **Retrieval-Augmented Generation (RAG):** Generates grounded AI answers, document summaries, and inline citations based on retrieved context.
* **Multimodal Search:** Uses CLIP (`clip-ViT-B-32`) to map both images and text into the same vector space, allowing users to search the text dataset using image queries.
* **Evaluation Suite:** Built-in tools to benchmark Search Engine performance (Precision@k, Recall@k, F1 Score).

## 🚀 Installation & Setup

This project uses `uv` for lightning-fast Python dependency management.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/rag-search-engine.git](https://github.com/yourusername/rag-search-engine.git)
   cd rag-search-engine

```

2. **Install dependencies:**
```bash
uv sync

```


3. **Set up Environment Variables:**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here

```



---

## 💻 Command Line Interfaces (CLI)

The project is structured into multiple CLI entry points, each demonstrating a different tier of the search architecture.

### 1. Keyword Search (`keyword_search_cli.py`)

Handles exact-match querying, tokenization, and BM25 scoring.

* `build` - Parse the dataset and build the inverted index.
* `tf <doc_id> <term>` - Calculate standard Term Frequency.
* `idf <term>` - Calculate Inverse Document Frequency.
* `tfidf <doc_id> <term>` - Calculate combined TF-IDF score.
* `bm25tf <doc_id> <term> [k1] [b]` - Calculate BM25 Term Frequency.
* `bm25idf <term>` - Calculate BM25 IDF score.
* `bm25search <query> [limit]` - Search the database using the advanced BM25 ranking algorithm.

### 2. Semantic Search (`semantic_search_cli.py`)

Handles vector embeddings, cosine similarity, and chunking.

* `verify` - Verify the embedding model loads correctly.
* `embed_text <text>` - Generate vector embeddings for a given string.
* `chunk <text> [--chunk-size] [--overlap]` - Split text into fixed-size chunks.
* `semantic_chunk <text> [--max-chunk-size] [--overlap]` - Split text intelligently to preserve context.
* `embed_chunks` - Process dataset and cache chunked vector embeddings.
* `search <query> [--limit]` - Search using semantic vector similarity.
* `search_chunked <query> [--limit]` - Search against chunked document embeddings for higher precision.

### 3. Hybrid Search (`hybrid_search_cli.py`)

Combines BM25 keyword scores with Semantic cosine similarity scores.

* `normalize <scores>` - Normalize raw scores to a 0.0 - 1.0 scale using Min-Max scaling.
* `weighted-search <query> [--alpha] [--limit]` - Search using a weighted average (alpha controls BM25 weight).
* `rrf-search <query> [--k] [--limit] [--enhance] [--rerank-method] [--evaluate]` - Advanced search using Reciprocal Rank Fusion, with options for LLM query enhancement (spell/rewrite/expand) and secondary re-ranking.

### 4. Retrieval-Augmented Generation (`augmented_generation_cli.py`)

Uses the Gemini API to synthesize answers based on retrieved context.

* `rag <query>` - Execute a full RAG pipeline to generate an AI answer.
* `summarize <query> [--limit]` - Generate a concise AI summary of the retrieved results.
* `citations <query> [--limit]` - Answer the query with inline citations referencing the source documents.
* `question <query> [--limit]` - Engage in an interactive Q&A session grounded in the search results.

### 5. Multimodal Search (`multimodal_search_cli.py`)

Embeds and searches across images and text using the `jina-clip-v2` or `clip-ViT-B-32` models.

* `verify_image_embedding <image_path>` - Generate and check the dimensional shape of an image embedding.
* `image_search <image_path>` - Upload a `.jpg`/`.png` to search the movie text dataset using mathematical visual similarity.

### 6. Evaluation (`evaluation_cli.py`)

Benchmarks the accuracy of the search engine.

* `--limit <k>` - Evaluate Precision@k and Recall@k against a golden dataset to track search performance improvements over time.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Package Manager:** `uv`
* **Embedding Models:** `sentence-transformers`, `transformers`, `CLIP`
* **Local Vectors:** `numpy`, `PIL`
* **LLM Integration:** `google-genai` (Gemini)

## 🎓 Acknowledgements

Huge thanks to **Isaac Flath** and the team at [Boot.dev](https://boot.dev) for creating the incredible curriculum that guided the construction of this engine.

```
http://googleusercontent.com/youtube_content/1

```