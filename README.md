# Simple RAG Pipeline with Haystack 2.0

This repository contains a modular implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using **Haystack 2.0**. It indexes Wikipedia articles regarding the *Seven Wonders of the Ancient World* and serves as an intelligent Q&A system that can answer questions using real factual context.

To keep things accessible, local, and completely free, this project runs on-device using Hugging Face's open-weight model (`Qwen/Qwen3-4B-Instruct-2507`) via `TransformersChatGenerator`.

---

## ⚙️ How It Works

This application follows a standard RAG pattern, split into two phases:

1. **Indexing Phase:**
   - Downloads the *Seven Wonders* dataset from Hugging Face.
   - Uses `SentenceTransformersDocumentEmbedder` to generate high-dimensional vector embeddings for the text documents.
   - Stores the documents and their mathematical embeddings in an ephemeral `InMemoryDocumentStore`.

2. **Query Phase:**
   - Converts the user's plain-text question into an embedding using `SentenceTransformersTextEmbedder`.
   - Queries the Document Store using an `InMemoryEmbeddingRetriever` to pull contextually matching Wikipedia articles.
   - Uses `ChatPromptBuilder` to structure a clear, context-rich prompt for the LLM.
   - Generates a grounded, hallucination-free answer using `TransformersChatGenerator`.

---

## 🛠️ Tech Stack & Components

- **Orchestration:** Haystack 2.0
- **Document Store:** `InMemoryDocumentStore` (simple, fast, local)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model:** `Qwen/Qwen3-4B-Instruct-2507` (running locally via Hugging Face Transformers)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Clone this repository, navigate to the directory, and install the required dependencies using pip:

```bash
pip install -r requirements.txt
