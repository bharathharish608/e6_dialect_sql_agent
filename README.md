# e6data Conversational SQL Agent

## Overview

**e6data Conversational SQL Agent** is a robust, production-grade, multi-turn conversational agent designed to answer questions about the e6data platform and SQL dialect, grounded in your official documentation. It features advanced memory, coreference resolution, custom entity tracking, and full audit logging, making it suitable for enterprise and compliance-critical environments.

---

## Product Purpose

- **Documentation-grounded Q&A:** Answers are always based on your official documentation, never hallucinated.
- **SQL and platform expertise:** Handles SQL, error diagnosis, deployment, and product-specific queries.
- **Auditability:** Every tool call, retrieval, and LLM response is logged for compliance and debugging.
- **Extensible:** Modular design for easy integration of new tools, memories, or LLMs.

---

## Project Structure

```
e6_dialect_sql/
├── src/
│   ├── main_agent.py      # Main conversational agent (entry point)
│   ├── ingest.py          # Documentation crawler and ingestion
│   ├── chunk.py           # Document chunking for retrieval
│   ├── embed.py           # Embedding and vector DB population
│   └── inspect_docs.py    # (Utility) Inspect docs/chunks
├── data/
│   ├── docs.json          # Raw crawled documentation
│   ├── metadata.json      # Crawl metadata
│   ├── chunks.json        # Chunked docs for retrieval
│   ├── chroma_db/         # ChromaDB persistent vector store
│   └── FAISS/
│       ├── entity_memory.jsonl  # Entity memory log (JSONL)
│       └── entity_memory.index  # FAISS index for entity memory
├── application_logs/      # Detailed technical logs (per conversation)
├── conversation_logs/     # Human-readable conversation logs
├── requirements.txt       # Python dependencies
├── config.env             # API keys and environment variables
└── README.md              # (You are here)
```

---

## Data Preparation Pipeline

The agent is grounded in your documentation, which must be ingested, chunked, and embedded before the agent can answer questions. This pipeline is **one-time** (or as-needed) and consists of three main steps:

### 1. Documentation Ingestion (`src/ingest.py`)
- **Purpose:** Crawl all internal documentation pages and extract their main content.
- **Key Function:** `ingest_docs(force=False)`
  - Crawls from a root URL, follows internal links, and extracts main/article/largest-div/body text.
  - Saves all docs to `data/docs.json` and crawl metadata to `data/metadata.json`.
- **How to run:**
  ```bash
  python src/ingest.py
  ```

### 2. Document Chunking (`src/chunk.py`)
- **Purpose:** Split each documentation page into smaller, semantically meaningful chunks for retrieval.
- **Key Function:** `chunk_and_persist_docs()`
  - Loads `data/docs.json`, splits each doc using LlamaIndex's `SimpleNodeParser`.
  - Each chunk is associated with its source URL.
  - Saves all chunks to `data/chunks.json`.
- **How to run:**
  ```bash
  python src/chunk.py
  ```

### 3. Embedding & Vector DB Population (`src/embed.py`)
- **Purpose:** Embed all chunks and store them in a persistent vector database (ChromaDB) for fast semantic search.
- **Key Steps:**
  - Loads `data/chunks.json`.
  - Optionally adds Apache Calcite SQL idioms as extra chunks for SQL reasoning.
  - Embeds all chunks using Sentence Transformers (`all-MiniLM-L6-v2`).
  - Stores embeddings, texts, and metadata in ChromaDB (`data/chroma_db/`).
- **How to run:**
  ```bash
  python src/embed.py
  ```

**Result:**
- Your documentation is now ready for retrieval-augmented generation (RAG) by the agent.
- All subsequent agent queries are grounded in this indexed knowledge base.

---

## Full System Flow

1. **Data Preparation (one-time or as-needed):**
   - `ingest.py` → `chunk.py` → `embed.py`
2. **Agent Interaction (every user session):**
   1. User enters a question or SQL-related query.
   2. Coreference resolution (Claude LLM) rewrites ambiguous references.
   3. Hybrid retrieval (`vector_search_tool`):
      - Semantic search (ChromaDB) + BM25 keyword search over chunks.
      - Merges, deduplicates, and returns top-k relevant docs.
   4. Memory update:
      - ConversationBufferMemory & ConversationSummaryMemory (LangChain)
      - CustomEntityMemory (FAISS + JSONL) for SQL/tables/columns/errors
   5. Entity extraction from user/agent turns.
   6. Prompt construction:
      - Includes retrieved docs, memory digests, entity digest, and resolved input.
   7. LLM call (Claude via LangChain):
      - Generates a grounded, cited answer.
   8. Logging:
      - Application log (detailed, technical)
      - Conversation log (clean, readable)

---

## Key Components & Classes

### `main_agent.py`
- **Main conversational loop:** Handles user input, memory, retrieval, prompt construction, LLM call, and logging.
- **`vector_search_tool(query, top_k)`:** Hybrid retrieval (ChromaDB + BM25) for documentation grounding.
- **`resolve_coreferences_with_llm(...)`:** Uses Claude to rewrite ambiguous user queries.
- **`CustomEntityMemory`:**
  - Stores and retrieves entity digests using FAISS and a JSONL log.
  - Files: `data/FAISS/entity_memory.index`, `data/FAISS/entity_memory.jsonl`
- **Memory:**
  - `ConversationBufferMemory` and `ConversationSummaryMemory` (LangChain)
- **Logging:**
  - `application_logs/` (detailed, technical)
  - `conversation_logs/` (clean, readable)

### `ingest.py` (Documentation Crawler)
- **`ingest_docs(force=False)`**: Crawls all internal documentation links from a root URL, extracts main content, and saves to `data/docs.json`.
  - Uses BeautifulSoup to parse HTML and extract the most relevant content from each page.
  - Handles navigation, error cases, and crawl deduplication.
  - Stores crawl metadata (timestamp, URL count, etc.) in `data/metadata.json`.

### `chunk.py` (Document Chunker)
- **`chunk_and_persist_docs()`**: Splits each documentation page into smaller, semantically meaningful chunks using LlamaIndex's `SimpleNodeParser`.
  - Each chunk is associated with its source URL for traceability.
  - Saves all chunks to `data/chunks.json`.

### `embed.py` (Embedding Pipeline)
- **Purpose:** Embeds all documentation chunks and stores them in ChromaDB for fast semantic retrieval.
- **Key Steps:**
  - Loads `data/chunks.json`.
  - Optionally adds Apache Calcite SQL idioms as extra chunks for SQL reasoning.
  - Embeds all chunks using Sentence Transformers (`all-MiniLM-L6-v2`).
  - Stores embeddings, texts, and metadata in ChromaDB (`data/chroma_db/`).

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd e6_dialect_sql
   ```
2. **Create and activate a Python 3.11+ virtual environment:**
   ```bash
   python3.11 -m venv venv311
   source venv311/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure API keys:**
   - Copy `config.env.example` to `.env` and fill in your Anthropic API key:
     ```
     cp config.env.example .env
     # Edit .env and set ANTHROPIC_API_KEY=...
     ```
5. **Ingest and prepare documentation:**
   ```bash
   python src/ingest.py
   python src/chunk.py
   python src/embed.py
   ```

---

## Running the Agent

```bash
source venv311/bin/activate
python src/main_agent.py
```
- The agent will prompt for user input in a conversational loop.
- All logs are written to `application_logs/` and `conversation_logs/`.

---

## Data & Storage
- **Documentation:** `data/docs.json`, `data/chunks.json`
- **ChromaDB vector store:** `data/chroma_db/`
- **Entity memory:** `data/FAISS/entity_memory.index`, `data/FAISS/entity_memory.jsonl`
- **Logs:** `application_logs/`, `conversation_logs/`

---

## Extending the Agent
- **Add new tools:** Implement and register new retrieval or reasoning tools in `main_agent.py`.
- **Swap LLMs:** Replace the Claude LLM integration with any LangChain-compatible LLM.
- **Custom memories:** Extend or modify `CustomEntityMemory` for new entity types.
- **Prompt engineering:** Adjust prompt construction for different reasoning or compliance needs.

---

## Requirements

See `requirements.txt`:
```
langchain
chromadb
requests
beautifulsoup4
python-dotenv
anthropic
sentence-transformers
langgraph>=0.5.1
rank_bm25
typing-extensions
tqdm
rich
faiss-cpu
langchain-anthropic
```

---

## Environment Variables

- `ANTHROPIC_API_KEY` (required): Your Anthropic Claude API key. Set in `.env`.

---

## License

Proprietary. For internal use at e6data or by authorized users only. 