# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Retrieval-Augmented Generation (RAG) web app that answers questions about course
materials. Tool-calling RAG: Claude decides whether to search the vector store rather
than retrieval happening on every query. FastAPI backend + vanilla JS frontend,
ChromaDB for vectors, Anthropic Claude for generation.

## Commands

All Python is run through `uv`. Requires Python 3.13.

```bash
uv sync                 # install/update dependencies from pyproject.toml + uv.lock
./run.sh                # start the server (chdir to backend, uvicorn with --reload, port 8000)

# Manual start (equivalent to run.sh):
cd backend && uv run uvicorn app:app --reload --port 8000

# Code quality:
./scripts/format.sh     # auto-format all Python with black
./scripts/check.sh      # black --check (formatting) + pytest; fails if unformatted
uv run pytest           # tests only
```

- Web UI: http://localhost:8000  •  API docs: http://localhost:8000/docs
- Requires `ANTHROPIC_API_KEY` in a `.env` at the repo root (`cp .env.example .env`).
  Loaded by `backend/config.py` via `load_dotenv()`. Server boots without it but every
  query fails.
- Code is formatted with **black** (config in `pyproject.toml`, `[tool.black]`). Run
  `./scripts/format.sh` after editing Python; `./scripts/check.sh` enforces it.
- There is **no build step**. `main.py` is an unused stub — not the entrypoint.

## Architecture

Request flow for a query (`frontend/script.js` → `backend/`):

```
POST /api/query (app.py)
  → RAGSystem.query (rag_system.py)        orchestrator; wraps query, pulls session history
    → AIGenerator.generate_response        1st Claude call, advertises search tool
      → [if stop_reason == tool_use]
        → ToolManager → CourseSearchTool → VectorStore.search   (search_tools.py, vector_store.py)
      → 2nd Claude call (no tools)          synthesizes final answer from tool results
  → collect sources, persist exchange, return {answer, sources, session_id}
```

Key cross-file concepts that aren't obvious from a single file:

- **Two ChromaDB collections** (`vector_store.py`): `course_catalog` (one doc per course,
  used to fuzzy-resolve a partial course name to its exact title) and `course_content`
  (the chunked material that's actually searched). A query with a `course_name` filter
  first resolves the name against `course_catalog`, then filters `course_content`.

- **Course title is the primary key** everywhere (Chroma IDs, dedup, links). Ingestion is
  idempotent on title only — re-ingesting requires a title change or clearing the DB.

- **Sources are side-state**, not return values. `CourseSearchTool` stashes them on
  `self.last_sources` during `execute`; `RAGSystem.query` reads them via
  `ToolManager.get_last_sources()` then calls `reset_sources()`. Not concurrency-safe.

- **Single tool round only.** `AIGenerator._handle_tool_execution` runs one round of tool
  calls then makes a final call *without* tools — Claude cannot chain a second search.
  The system prompt in `ai_generator.py` also enforces "one search per query".

- **Sessions are in-memory** (`session_manager.py`); restarting the server drops all
  history. History is truncated to `MAX_HISTORY` exchanges.

- **Startup ingestion**: `app.py`'s startup hook loads `docs/` into ChromaDB (creating
  `backend/chroma_db/`) and downloads the embedding model on first run. Already-ingested
  courses (by title) are skipped.

## Adding course documents

`docs/` holds plain-text course scripts parsed by `document_processor.py`. Expected format:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson body...>

Lesson 1: ...
```

Text is chunked by **sentence** (not raw chars) up to `CHUNK_SIZE` with `CHUNK_OVERLAP`
backtracking whole sentences. Note: chunk context-prefixing is inconsistent between the
mid-document lesson loop and the final-lesson block — match the existing path you edit.

## Config

All tunables live in `backend/config.py` (`Config` dataclass): `ANTHROPIC_MODEL`,
`EMBEDDING_MODEL`, `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100), `MAX_RESULTS` (5),
`MAX_HISTORY` (2), `CHROMA_PATH`.

## Dev-only notes

- **No-cache static files** — the frontend is mounted with `DevStaticFiles` (in
  `backend/app.py`), which sends `Cache-Control: no-cache` so edits to `script.js` /
  `style.css` appear without a hard browser refresh. This is a development
  convenience only. **Before any real deployment, revert the mount to plain
  `StaticFiles`** so browsers can cache static assets normally.
