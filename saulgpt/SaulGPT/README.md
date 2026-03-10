# SaulGPT

SaulGPT is a local-first legal AI assistant for Indian law. It combines a Next.js chat UI, a FastAPI backend, a pgvector-backed retrieval layer, and a local Ollama-served Llama 3.1 model to answer legal questions, explain statutory provisions, and generate limited legal drafting templates with citations.

This file is the main developer handbook for the repo. It consolidates the useful project-level information that was previously spread across the root README, `ml/README.md`, `data/DATA_SOURCES.md`, and the implementation itself.

## Why This Project Exists

Indian legal research is difficult for three reasons:

- statutory text is large and spread across multiple codes
- practitioners often need exact sections, not generic summaries
- new criminal laws such as BNS, BNSS, and BSA have replaced the older IPC / CrPC / Evidence Act structure, so cross-code understanding matters

SaulGPT tries to solve that by:

- grounding answers in retrieved statutory text instead of relying only on model memory
- keeping the stack local for privacy-sensitive legal work
- supporting both legal lookup and limited legal drafting help
- exposing the retrieval path clearly through citations

## What The App Does Today

- answers legal questions about BNS, BNSS, BSA, IPC, and the Constitution of India
- retrieves relevant chunks from pgvector using exact match, keyword search, and vector similarity
- streams answers to the UI with citations
- handles legal drafting requests with constrained template-style output
- routes greetings and non-legal prompts away from the RAG pipeline

Important limitations:

- it is not a full legal reasoning engine and should not be treated as legal advice
- drafting is intentionally conservative and uses placeholders when retrieval lacks procedural context
- the quality of answers depends heavily on the quality of `data/processed/chunks.jsonl`

## High-Level Architecture

```text
Browser UI (Next.js 16 + React 19)
        |
        |  streaming chat via Vercel AI SDK-compatible data stream
        v
Backend API (FastAPI)
        |
        |-- route greeting / meta / out-of-scope prompts directly
        |
        |-- route legal prompts into RAG pipeline
        v
RAG Layer (LangChain + custom retrieval logic)
        |
        |-- exact section matching
        |-- keyword search
        |-- vector similarity search
        v
pgvector (PostgreSQL) <---- embeddings from BAAI/bge-base-en-v1.5
        |
        v
Ollama local model serving (`saulgpt` -> Llama 3.1 8B base / customized)
```

## Tech Stack And Why It Was Chosen

### Frontend

- `Next.js 16` + `React 19`: app shell and UI rendering
- `@ai-sdk/react` + `ai`: chat state management and streaming protocol handling
- `Tailwind CSS 4`: fast iteration for the split-pane legal UI
- `react-markdown` + `remark-gfm`: render model output and document previews cleanly
- `lucide-react`: consistent icon set

Why: fast local development, simple streaming UX, and enough flexibility to build a legal research UI without a heavy custom state layer.

### Backend

- `FastAPI`: simple, fast HTTP API for `/api/health`, `/api/search`, `/api/chat`
- `LangChain` / `langchain_classic`: retrieval chain orchestration
- `langchain-ollama`: local model integration via Ollama
- `psycopg2-binary`: direct PostgreSQL access for retrieval and ingestion
- `pydantic-settings`: `.env`-backed configuration
- `ai-datastream`: Vercel AI SDK-compatible stream parts for chat output

Why: good fit for a small local RAG service with explicit control over retrieval and streaming behavior.

### Retrieval / ML

- `pgvector` on PostgreSQL: vector storage plus structured metadata queries in the same database
- `sentence-transformers`: embedding generation
- `BAAI/bge-base-en-v1.5`: embedding model used for indexing and semantic search
- `Ollama`: local model serving with no external API dependency
- `Llama 3.1 8B q4_K_M`: local model base used for SaulGPT

Why: this combination is lightweight enough to run locally while still supporting legal-text retrieval and answer generation.

## Repository Map

```text
.
├── README.md                     # Main handbook (this file)
├── DEMO_SCRIPT.md                # Hackathon/demo flow
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app entrypoint
│   │   ├── api/
│   │   │   ├── router.py         # Mounts chat, health, and search routes
│   │   │   └── endpoints/
│   │   │       ├── chat.py       # Streaming chat endpoint + intent routing
│   │   │       ├── search.py     # Raw retrieval endpoint
│   │   │       └── health.py     # Ollama / pgvector health check
│   │   ├── core/config.py        # Settings loaded from `.env`
│   │   ├── dependencies.py       # LLM, vector store, and RAG chain wiring
│   │   ├── models/schemas.py     # Request/response models
│   │   └── services/
│   │       ├── rag_service.py    # Prompting and retrieval chain definition
│   │       └── vector_store.py   # Search heuristics + vector retrieval
│   ├── scripts/ingest.py         # Rebuilds pgvector from processed chunks
│   ├── requirements.txt
│   ├── tests/test_edge_cases.py
│   └── .env                      # Local runtime settings
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/page.tsx          # Main split-pane SaulGPT screen
│       ├── hooks/use-legal-chat.ts
│       ├── components/chat/      # Chat list, input, citations, document preview
│       ├── components/legal-disclaimer.tsx
│       └── lib/legal/format.ts   # Legal document detection / formatting helpers
├── data/
│   ├── DATA_SOURCES.md           # Corpus source details
│   ├── raw/                      # Source legal texts (gitignored)
│   └── processed/chunks.jsonl    # Canonical ingestion input
├── docker/
│   ├── docker-compose.yml        # PostgreSQL + pgvector service
│   └── init.sql                  # DB init
└── ml/
    ├── Modelfile                 # Ollama model customization
    ├── finetune_saulgpt.ipynb    # Unsloth + QLoRA workflow
    └── README.md                 # Legacy ML notes; useful but not source of truth
```

Note: `frontend/README.md` is standard create-next-app boilerplate and is not the canonical documentation for this project.

## End-To-End Request Flow

### Chat request path

1. The user types into the frontend chat UI.
2. `frontend/src/hooks/use-legal-chat.ts` sends the request to `/api/chat` and transforms the backend stream into AI SDK-compatible parts.
3. `backend/app/api/endpoints/chat.py` inspects the last user message.
4. The backend router chooses one of these paths:
   - greeting -> direct short legal-assistant intro
   - meta/help -> direct capability response
   - non-legal -> polite refusal
   - legal query -> RAG pipeline
5. For legal queries, `get_rag_chain()` builds a retrieval chain.
6. `backend/app/services/vector_store.py` searches pgvector.
7. `backend/app/services/rag_service.py` builds the grounded prompt and sends it to Ollama.
8. The backend streams text chunks and citation parts back to the frontend.
9. The frontend renders the answer, citations, and document preview if the answer looks like a legal draft.

### Search request path

`/api/search` exposes retrieval results directly. This is useful for debugging retrieval quality without involving generation.

## What "RAG" Means In This Repo

In SaulGPT, RAG is not a generic buzzword. It means:

- legal text is chunked into `data/processed/chunks.jsonl`
- each chunk is embedded with `BAAI/bge-base-en-v1.5`
- embeddings plus chunk metadata are stored in `legal_document_chunks`
- user queries are searched against this corpus before generation
- retrieved chunks are inserted into the prompt with metadata such as act, section, title, and source
- the model answers using the retrieved legal text as the primary authority

The current answer policy is hybrid-grounded:

- retrieved context is the primary authority
- the model may use its internal knowledge only to improve explanation and structure
- the model must not contradict retrieved context
- if something is not supported by retrieved context, it should be framed as general guidance requiring verification

## Retrieval Strategy In Detail

`backend/app/services/vector_store.py` implements hybrid retrieval, not just pure vector search.

It does all of the following:

- extracts mentioned acts such as `BNS`, `BNSS`, `BSA`, `IPC`, `Constitution`
- extracts section/article references like `Section 103` or `Article 21`
- boosts procedural queries toward BNSS, especially FIR-related queries
- attempts exact section retrieval first when possible
- uses keyword scoring on titles and content
- falls back to vector similarity search when exact/keyword matching is insufficient

This is why the README should not describe SaulGPT as "just embeddings plus LLM". It has statute-aware routing logic on top of semantic retrieval.

## Legal Data In The Repo

### Raw corpus

`data/raw/` is gitignored and holds the source legal texts.

Current documented sources include:

- IPC JSON dataset
- Constitution of India JSON dataset
- BNS 2023 text
- BNSS 2023 text
- BSA 2023 text
- Aalap instruction dataset (for fine-tuning; gated external dataset)

See `data/DATA_SOURCES.md` for schema and provenance details.

### Processed corpus

`data/processed/chunks.jsonl` is the canonical ingestion input. Each line is a JSON object containing:

- `content`: chunk text
- `metadata`: fields such as `act`, `section`, `title`, `source`, `chunk_index`

### Vector database

The PostgreSQL table is `legal_document_chunks`.

It stores:

- chunk text
- JSONB metadata
- embedding vector

## How Ingestion Works

`backend/scripts/ingest.py` is the rebuild script for the vector store.

Important behavior:

- it reads `data/processed/chunks.jsonl`
- it loads the embedding model through `app.services.vector_store.get_embedding_model()`
- it computes embeddings in batches
- it runs `TRUNCATE legal_document_chunks RESTART IDENTITY`
- it reinserts the entire processed corpus from scratch

This means ingestion is a rebuild, not an append.

If you improve chunking or add new statutes, regenerate `chunks.jsonl` and rerun ingest.

## Running The Full System Locally

### 1. Start pgvector

```bash
cd /Users/aetherion/Documents/munna-work
docker compose -f docker/docker-compose.yml up -d
```

### 2. Set up the backend

```bash
cd /Users/aetherion/Documents/munna-work/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure backend settings

Create or update `backend/.env`:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=saulgpt
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/saulgpt
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### 4. Prepare the local model

Install and start Ollama if needed:

```bash
brew install ollama
brew services start ollama
```

Pull the base model and create the local SaulGPT variant:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
cd /Users/aetherion/Documents/munna-work/ml
ollama create saulgpt -f Modelfile
```

### 5. Build the vector index

```bash
cd /Users/aetherion/Documents/munna-work/backend
venv/bin/python scripts/ingest.py
```

### 6. Start the backend

```bash
cd /Users/aetherion/Documents/munna-work/backend
source venv/bin/activate
uvicorn app.main:app --port 8000
```

### 7. Start the frontend

Use Bun, not npm:

```bash
cd /Users/aetherion/Documents/munna-work/frontend
bun install
bun run dev
```

Open `http://localhost:3000`.

## Useful Manual Test Prompts

Use these to quickly validate behavior:

- `hi`
- `What is Section 103 BNS?`
- `Explain Section 103 BNS in simple terms`
- `What is the procedure to file an FIR under BNS?`
- `What is the equivalent of IPC Section 302 under BNS?`
- `Draft a bail application for theft under Section 303 BNS`
- `tell me a joke`

Expected behavior:

- greetings -> direct legal-assistant response, no RAG
- non-legal chat -> refusal
- legal questions -> grounded RAG answer with citations
- drafting -> constrained template output with placeholders where context is missing

## API Overview

### `GET /api/health`

Checks whether Ollama and pgvector are reachable.

Example:

```bash
curl http://localhost:8000/api/health
```

### `POST /api/search`

Returns raw retrieval results from the vector store.

Example:

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Section 103 BNS?","top_k":5}'
```

Use this when you want to debug retrieval separately from generation.

### `POST /api/chat`

Streaming endpoint used by the frontend.

Example:

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is Section 103 BNS?"}]}'
```

The backend emits text parts and citation/error data parts in a Vercel AI SDK-compatible stream.

## How The Local Model Is Configured

There are two major places that shape SaulGPT behavior.

### 1. Runtime model selection

`backend/app/core/config.py` loads settings from `backend/.env`.

Right now:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=saulgpt`
- `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`

### 2. Ollama model customization

`ml/Modelfile` defines the local `saulgpt` model on top of `llama3.1:8b-instruct-q4_K_M`.

This is where you customize:

- system behavior at the base model layer
- parameters such as `temperature` and `top_p`
- stop sequences

After changing `ml/Modelfile`, recreate the model:

```bash
cd /Users/aetherion/Documents/munna-work/ml
ollama create saulgpt -f Modelfile
```

## How To Add More Domain Detail Without Fine-Tuning

If you want SaulGPT to know more, there are three different levers. They are not the same.

### Option 1: Prompt customization

Change:

- `backend/app/services/rag_service.py` -> `SYSTEM_PROMPT`
- `ml/Modelfile` -> base Ollama system prompt

Use this when you want to change:

- tone
- formatting rules
- safety/refusal policy
- acronym handling
- how the model should treat retrieved context vs general knowledge

This is fast and cheap. It does not add new legal knowledge by itself.

### Option 2: RAG corpus update

Change:

- files in `data/raw/`
- preprocessing output in `data/processed/chunks.jsonl`
- rerun `backend/scripts/ingest.py`

Use this when you want SaulGPT to answer from newer laws, better chunking, more acts, cleaner metadata, or better retrieval coverage.

This is the best choice when the problem is missing legal source material.

### Option 3: Fine-tuning

Use `ml/finetune_saulgpt.ipynb` when you want to change the model's learned behavior or domain-specific generation patterns more deeply.

Use this when prompt changes are not enough and retrieval alone cannot solve the issue.

## How To Update The RAG Corpus

Typical workflow:

1. Add or update source legal files in `data/raw/`
2. Update your preprocessing logic if the source format changed
3. Regenerate `data/processed/chunks.jsonl`
4. Re-run ingestion:

```bash
cd /Users/aetherion/Documents/munna-work/backend
venv/bin/python scripts/ingest.py
```

5. Sanity check the database:

```bash
docker exec saulgpt-pgvector psql -U postgres -d saulgpt -c "SELECT count(*) FROM legal_document_chunks;"
```

Useful mental model:

- update raw data -> when source material changed
- update `chunks.jsonl` -> when chunking / metadata changed
- rerun ingest -> when you want pgvector to reflect those changes

## How Fine-Tuning Works In This Repo

The notebook is `ml/finetune_saulgpt.ipynb`.

Based on the notebook contents, the workflow is:

1. install Unsloth and dependencies in Colab
2. load `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
3. attach LoRA adapters for QLoRA training
4. load the Aalap instruction dataset if available
5. fall back to a manually built dataset if Aalap is unavailable
6. fine-tune with `SFTTrainer`
7. export the result to GGUF
8. bring the GGUF back to local and use it with Ollama

Notebook facts already visible in the repo:

- framework: Unsloth
- training style: QLoRA / PEFT
- preferred dataset: `opennyaiorg/aalap_instruction_dataset`
- fallback dataset path exists in the notebook
- export target: GGUF

Important distinction:

- prompt customization changes instructions
- RAG updates change what legal source material can be retrieved
- fine-tuning changes model weights

Do the simplest thing that solves your problem.

## Current Data Sources

From `data/DATA_SOURCES.md`, the documented sources include:

- IPC JSON dataset
- Constitution of India JSON dataset
- BNS text
- BNSS text
- BSA text
- Aalap legal instruction dataset for fine-tuning

This matters because retrieval quality depends on both source coverage and chunk quality. If SaulGPT gives weak answers, inspect the underlying corpus before assuming the model is the problem.

## Common Pitfalls

### 1. You changed the raw data but answers did not improve

You probably forgot to regenerate `data/processed/chunks.jsonl` or rerun ingest.

### 2. You changed `ml/Modelfile` but behavior did not change

You must recreate the Ollama model with `ollama create saulgpt -f Modelfile`.

### 3. The backend is up but retrieval seems wrong

Check `/api/search` directly before debugging prompt quality.

### 4. The model answers too generally

Inspect:

- retrieval quality in `vector_store.py`
- prompt policy in `rag_service.py`
- whether the relevant source text actually exists in `chunks.jsonl`

### 5. You want better answers for one specific domain

Decide first whether you need:

- better prompt rules
- better retrieved legal text
- actual fine-tuning

They solve different problems.

### 6. Very long user prompts behave strangely

`chat.py` truncates queries above 2000 characters and reduces retrieval depth for very long prompts.

## Suggested Onboarding Path For A New Developer

If you are new to the repo, read things in this order:

1. this `README.md`
2. `backend/app/api/endpoints/chat.py`
3. `backend/app/services/vector_store.py`
4. `backend/app/services/rag_service.py`
5. `backend/scripts/ingest.py`
6. `frontend/src/hooks/use-legal-chat.ts`
7. `frontend/src/app/page.tsx`
8. `data/DATA_SOURCES.md`
9. `ml/Modelfile`
10. `ml/finetune_saulgpt.ipynb`

That sequence mirrors the actual product path: request -> routing -> retrieval -> prompting -> data pipeline -> frontend stream handling -> model customization.

## Developer Workflow Tips

- use `bun` in the frontend
- use the backend virtualenv in `backend/venv`
- verify retrieval changes with `/api/search` before judging generation
- verify UX changes with both `bun run lint` and `bun run build`
- keep `README.md` as the source of truth instead of adding scattered mini-docs unless a new document genuinely needs its own lifecycle

## Quick Commands Reference

```bash
# Start pgvector
docker compose -f docker/docker-compose.yml up -d

# Backend install
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Rebuild vector index
cd backend && venv/bin/python scripts/ingest.py

# Start backend
cd backend && source venv/bin/activate && uvicorn app.main:app --port 8000

# Start frontend
cd frontend && bun install && bun run dev

# Recreate SaulGPT model after Modelfile changes
cd ml && ollama create saulgpt -f Modelfile

# Health check
curl http://localhost:8000/api/health

# Debug retrieval
curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d '{"query":"What is Section 103 BNS?"}'

# Debug streaming chat
curl -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is Section 103 BNS?"}]}'
```

If you keep this mental model clear - prompt rules vs retrieval data vs model weights - the codebase becomes much easier to work with.
# Saul
