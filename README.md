# 🚀 Professional RAG System & Workshop

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge)
![Coverage](https://img.shields.io/badge/Coverage-80%25-brightgreen?style=for-the-badge)

A modular, production-ready **Retrieval-Augmented Generation (RAG)** system built with Python, LangChain, and Groq. This repository serves as both a reference architecture for enterprise-grade RAG applications and a comprehensive educational workshop.

---

## ✨ Features

### 🧠 Core Intelligence

- **Advanced Retrieval**: Triple-strategy retrieval combining **Semantic Search**, **BM25 (Keyword)**, and **HyDE (Hypothetical Document Embeddings)**.
- **Reranking**: Precision filtering using **FlashRank** (Cross-Encoder) to reorder candidates.
- **Semantic Caching**: 2-level cache with **Intent Normalization** to save costs and reduce latency.
- **Smart Chunking**: Configurable strategies:
  - **Semantic Chunking**: Splits text based on meaning shifts.
  - **Recursive Chunking**: Traditional size-based splitting.

### 🏗️ Architecture

- **Clean Layout**: `src/` based structure with isolated modules (Ingestion, Retrieval, Chain, Evaluation).
- **Local Embeddings**: Runs `paraphrase-multilingual-MiniLM-L12-v2` locally for privacy and speed.
- **Type Safety**: Full Python type hinting + Pydantic validation.
- **Observability**: Integrated with **LangSmith** for tracing.

### 🖥️ Interface

- **Streamlit UI**: ChatGPT-style interface with dark mode and sidebar configuration.
- **Dynamic Config**: Adjust chunking strategies and parameters directly from the UI.

---

## 🛠️ Tech Stack

| Component      | Technology                                                     | Description                          |
| -------------- | -------------------------------------------------------------- | ------------------------------------ |
| **LLM**        | [Groq](https://groq.com/)                                      | Llama 3.3 70B (Ultra-fast inference) |
| **Vector DB**  | [ChromaDB](https://www.trychroma.com/)                         | Local persistent vector storage      |
| **Embeddings** | [Sentence-Transformers](https://sbert.net/)                    | Multilingual local model             |
| **Reranker**   | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | Light-weight Cross-Encoder           |
| **Framework**  | [LangChain](https://www.langchain.com/)                        | Orchestration & Abstractions         |
| **UI**         | [Streamlit](https://streamlit.io/)                             | Web interface                        |
| **Manager**    | [uv](https://github.com/astral-sh/uv)                          | Fast Python package manager          |

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

### 2. Installation

Clone the repo and sync dependencies:

```bash
git clone https://github.com/yourusername/rag-workshop.git
cd rag-workshop
uv sync
```

### 3. Configuration

```env
# Required
GROQ_API_KEY=gsk_...

# Optional (for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=RAG_Workshop_Pro
```

### 4. Running the App

```bash
uv run streamlit run src/main.py
```

🧪 Testing & Evaluation

```bash
    uv run pytest
    uv run pytest --cov # with coverage
```

Run Retrieval Evaluation

```bash
    uv run evaluate.py
```

📂 Project Structure

```text
src/
└── rag/
    ├── chain/          # Conversational logic (Memory, Cache, Router)
    ├── ingestion/      # Loading & Chunking strategies
    ├── retrieval/      # Vector Store, Hybrid Search, Reranking
    ├── config.py       # Centralized settings
    ├── pipeline.py     # Main Orchestrator
    └── cli.py          # Streamlit Interface
```

## 📄 License

This project is licensed under the MIT License.
See the `LICENSE` file for details.

---
