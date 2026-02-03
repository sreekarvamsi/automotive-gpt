"""
Pydantic request / response schemas for the REST API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────

class VehicleFilter(BaseModel):
    """Optional metadata filters to narrow retrieval."""
    make: str | None = None
    model: str | None = None
    year: int | None = None
    subsystem: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID for multi-turn. Omit to start a new conversation.",
    )
    message: str = Field(..., min_length=1, description="The user's question.")
    filters: VehicleFilter | None = None


class Citation(BaseModel):
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation] = []
    timing: dict[str, Any] = {}
    model: str = ""


# ─────────────────────────────────────────────────────────────
# Conversations
# ─────────────────────────────────────────────────────────────

class MessageDetail(BaseModel):
    id: int
    role: str
    content: str
    created_at: str | None = None
    timing: dict[str, Any] | None = None


class ConversationResponse(BaseModel):
    id: str
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[MessageDetail] = []


class DeleteResponse(BaseModel):
    deleted: bool
    conversation_id: str


# ─────────────────────────────────────────────────────────────
# Vehicles
# ─────────────────────────────────────────────────────────────

class VehicleListResponse(BaseModel):
    sources: list[str]
    total: int


# ─────────────────────────────────────────────────────────────
# Ingestion (admin)
# ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to folder containing manuals (server-side).")
    vehicle_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Default metadata to stamp on all ingested chunks (make, model, year, subsystem).",
    )


class IngestResponse(BaseModel):
    status: str          # "success" | "error"
    chunks_indexed: int = 0
    message: str = ""


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str = ""
