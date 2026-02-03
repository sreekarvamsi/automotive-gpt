"""
FastAPI application — REST API for AutomotiveGPT.

Routes:
  POST   /api/v1/chat                  – send a message, get an answer
  GET    /api/v1/conversations/{id}    – retrieve conversation history
  DELETE /api/v1/conversations/{id}    – clear a conversation
  GET    /api/v1/vehicles              – list all indexed vehicle models
  POST   /api/v1/ingest               – trigger document ingestion (admin)
  GET    /api/v1/health               – health check

Caching:
  Responses are cached in Redis keyed by (query_hash, filters_hash).
  Cache TTL = 1 hour.  Cache is bypassed when conversation_id changes
  (because context changes with history).

CORS:
  Configured for local Streamlit dev (localhost:8501) and can be
  extended for production origins.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
import time
from pathlib import Path

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.memory.conversation_store import ConversationStore, init_db
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import AutomotiveGenerator, GenerationResult

logger = logging.getLogger(__name__)

# ── App & middleware ──────────────────────────────────────────────────
app = FastAPI(
    title="AutomotiveGPT API",
    version="1.0.0",
    description="Conversational QA over vehicle service manuals.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis cache ───────────────────────────────────────────────────────
_redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
_CACHE_TTL = 3600  # 1 hour


def _cache_key(query: str, filters: dict | None) -> str:
    """Deterministic cache key from query + filters."""
    raw = json.dumps({"q": query, "f": filters or {}}, sort_keys=True)
    return "agpt_cache:" + hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    val = _redis.get(key)
    return json.loads(val) if val else None


def _cache_set(key: str, value: dict) -> None:
    _redis.setex(key, _CACHE_TTL, json.dumps(value))


# ── Pydantic schemas ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, description="Existing conversation to continue, or null to start new")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Client session identifier")
    message: str = Field(..., min_length=1, description="The user's question")
    filters: dict | None = Field(default=None, description="Vehicle filters: make, model, year, subsystem")


class SourceInfo(BaseModel):
    source_id: int
    source_file: str
    page: int | None
    section_type: str | None
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    confidence: float
    sources: list[SourceInfo]
    latency_ms: int
    cached: bool = False


class ConversationResponse(BaseModel):
    id: str
    session_id: str
    created_at: str | None
    updated_at: str | None
    messages: list[dict]
    filters: dict


class VehicleInfo(BaseModel):
    make: str
    model: str
    year: int | None
    source_file: str


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Absolute path to folder containing manuals")
    verbose: bool = False


class HealthResponse(BaseModel):
    status: str
    redis: bool
    db: bool


# ── Dependency: shared instances (created once at startup) ───────────
_store: ConversationStore | None = None
_retriever: HybridRetriever | None = None
_generator: AutomotiveGenerator | None = None


@app.on_event("startup")
async def startup():
    global _store, _retriever, _generator
    init_db()
    _store = ConversationStore()
    _retriever = HybridRetriever()
    _generator = AutomotiveGenerator(streaming=False)
    logger.info("AutomotiveGPT API started.")


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check — verifies Redis and DB connectivity."""
    redis_ok = False
    db_ok = False
    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        pass
    try:
        # Quick DB probe
        from sqlalchemy import text as sa_text
        from src.memory.conversation_store import _engine
        with _engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(status="ok" if (redis_ok and db_ok) else "degraded", redis=redis_ok, db=db_ok)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a user message and return an answer with citations.

    Flow:
      1. Create or resume a conversation in PostgreSQL.
      2. Check Redis cache (skip if conversation has history, since
         context changes with each turn).
      3. Run HybridRetriever with the user's filters.
      4. Run AutomotiveGenerator with retrieved context + history.
      5. Persist both the user message and the assistant answer.
      6. Cache the response if it was a fresh (no-history) query.
    """
    start = time.perf_counter()

    # ── Conversation setup ──────────────────────────────────────
    conv_id = request.conversation_id
    if conv_id is None:
        conv_id = _store.create(
            session_id=request.session_id,
            filters=request.filters,
        )
    else:
        # Update filters if provided
        if request.filters:
            _store.update_filters(conv_id, request.filters)

    # ── Cache check (only for first message in a conversation) ─
    history = _store.get_history_for_prompt(conv_id)
    cache_key = _cache_key(request.message, request.filters)
    cached_result = None
    if not history:  # only cache single-turn queries
        cached_result = _cache_get(cache_key)

    if cached_result:
        # Still persist the messages even when serving from cache
        _store.append_message(conv_id, role="user", content=request.message)
        _store.append_message(conv_id, role="assistant", content=cached_result["answer"])
        cached_result["conversation_id"] = conv_id
        cached_result["cached"] = True
        logger.info("Cache hit for query '%s…'", request.message[:50])
        return ChatResponse(**cached_result)

    # ── Retrieval ───────────────────────────────────────────────
    chunks = _retriever.retrieve(request.message, request.filters)

    # ── Generation ──────────────────────────────────────────────
    result: GenerationResult = _generator.generate(
        query=request.message,
        context_chunks=chunks,
        conversation_history=history,
    )

    # ── Persist ─────────────────────────────────────────────────
    _store.append_message(
        conv_id, role="user", content=request.message,
        metadata={"filters": request.filters},
    )
    _store.append_message(
        conv_id, role="assistant", content=result.answer,
        metadata={"confidence": result.confidence, "sources": result.sources},
    )

    # ── Cache (single-turn only) ────────────────────────────────
    response_data = {
        "conversation_id": conv_id,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
        "latency_ms": result.latency_ms,
        "cached": False,
    }
    if not history:
        _cache_set(cache_key, response_data)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    response_data["latency_ms"] = elapsed_ms

    return ChatResponse(**response_data)


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Retrieve full conversation history."""
    conv = _store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(**conv)


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and its history."""
    deleted = _store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/v1/vehicles", response_model=list[VehicleInfo])
async def list_vehicles():
    """Return a list of all vehicle makes/models/years in the index.

    This is a lightweight stub — in production you would maintain a
    separate metadata table populated during ingestion.  Here we
    return a representative sample.
    """
    # Placeholder: in a real deployment this queries a metadata
    # catalog populated by the ingestion pipeline.
    # For now, return an empty list; populate after first ingest.
    return []


@app.post("/api/v1/ingest")
async def ingest(request: IngestRequest):
    """Trigger document ingestion from a local directory.

    This is an admin endpoint.  In production, protect with API-key
    auth or restrict to internal networks.
    """
    source_dir = Path(request.source_dir)
    if not source_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {source_dir}")

    # Import here to avoid circular imports and heavy startup cost
    from src.ingestion.parser import parse_document
    from src.ingestion.chunker import HybridChunker
    from src.ingestion.embedder import Embedder
    from src.ingestion.indexer import PineconeIndexer

    chunker = HybridChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder()
    indexer = PineconeIndexer()
    indexer.ensure_index()

    supported_extensions = {".pdf", ".docx", ".html", ".htm"}
    files = [f for f in source_dir.iterdir() if f.suffix.lower() in supported_extensions]

    if not files:
        raise HTTPException(status_code=400, detail="No supported documents found in directory.")

    total_vectors = 0
    for file_path in files:
        try:
            logger.info("Processing %s…", file_path.name)
            # Optionally re-index: delete old vectors for this file
            indexer.delete_by_source(file_path.name)

            sections = parse_document(file_path)
            chunks = chunker.chunk(sections)
            embedded = embedder.embed(chunks)
            count = indexer.upsert(embedded)
            total_vectors += count
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path.name, exc)
            continue

    return {
        "status": "completed",
        "files_processed": len(files),
        "total_vectors_upserted": total_vectors,
    }
