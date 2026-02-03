"""
PostgreSQL-backed conversation memory.

Schema (single table for simplicity):
  conversations
    id              – UUID primary key
    session_id      – caller-supplied session identifier
    created_at      – timestamp
    updated_at      – timestamp
    messages        – JSONB array of {role, content, metadata} dicts
    filters         – JSONB of the last-used vehicle filters

The `messages` column stores the full conversation as a flat list
so we can append in O(1) and replay the entire history in one query.

A `get_history_for_prompt()` helper returns only the (role, content)
pairs needed for the OpenAI messages list, trimmed to a max token
budget so we never blow the context window.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, Session as SASession
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from src.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# ── ORM model ─────────────────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc),
                        onupdate=datetime.now(timezone.utc))
    messages = Column(JSON, default=list)   # list of {role, content, metadata}
    filters = Column(JSON, default=dict)    # last-used vehicle filters


# ── Engine & session factory ──────────────────────────────────────────
_engine = create_engine(settings.postgres_url, pool_pre_ping=True)


def init_db() -> None:
    """Create tables if they don't exist.  Safe to call on every startup."""
    Base.metadata.create_all(_engine)
    logger.info("Database tables ensured.")


def _get_session() -> SASession:
    return SASession(_engine)


# ── CRUD layer ────────────────────────────────────────────────────────
class ConversationStore:
    """High-level interface for conversation persistence.

    Usage:
        store = ConversationStore()
        conv_id = store.create("session-abc")
        store.append_message(conv_id, role="user", content="Hello")
        history = store.get_history_for_prompt(conv_id)
    """

    # ── Create ──────────────────────────────────────────────────
    def create(self, session_id: str, filters: dict | None = None) -> str:
        """Create a new conversation. Returns the conversation UUID as a string."""
        conv_id = uuid.uuid4()
        with _get_session() as db:
            db.add(Conversation(
                id=conv_id,
                session_id=session_id,
                messages=[],
                filters=filters or {},
            ))
            db.commit()
        logger.info("Created conversation %s for session '%s'.", conv_id, session_id)
        return str(conv_id)

    # ── Append message ──────────────────────────────────────────
    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Append a single message to the conversation."""
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            if conv is None:
                raise ValueError(f"Conversation {conversation_id} not found.")

            messages = list(conv.messages)  # copy so SQLAlchemy detects the change
            messages.append({
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            conv.messages = messages
            db.commit()

    # ── Update filters ──────────────────────────────────────────
    def update_filters(self, conversation_id: str, filters: dict) -> None:
        """Store the current vehicle filters for this conversation."""
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            if conv is None:
                raise ValueError(f"Conversation {conversation_id} not found.")
            conv.filters = filters
            db.commit()

    # ── Read ────────────────────────────────────────────────────
    def get_conversation(self, conversation_id: str) -> dict | None:
        """Return the full conversation record as a dict, or None."""
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            if conv is None:
                return None
            return {
                "id": str(conv.id),
                "session_id": conv.session_id,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                "messages": conv.messages,
                "filters": conv.filters,
            }

    def get_history_for_prompt(
        self,
        conversation_id: str,
        max_messages: int = 20,
    ) -> list[dict]:
        """Return the last N messages as [{role, content}, …] for the OpenAI messages list.

        We cap at max_messages (default 20) to stay within the context
        window budget.  The most recent messages are kept.
        """
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            if conv is None:
                return []

            # Take the tail (most recent) messages
            recent = conv.messages[-max_messages:]
            return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_filters(self, conversation_id: str) -> dict:
        """Return the stored filters for a conversation."""
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            return conv.filters if conv else {}

    # ── Delete ──────────────────────────────────────────────────
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        with _get_session() as db:
            conv = db.query(Conversation).filter_by(id=conversation_id).one_or_none()
            if conv is None:
                return False
            db.delete(conv)
            db.commit()
        logger.info("Deleted conversation %s.", conversation_id)
        return True

    # ── List ────────────────────────────────────────────────────
    def list_conversations(self, session_id: str, limit: int = 50) -> list[dict]:
        """List conversations for a given session (most recent first)."""
        with _get_session() as db:
            convs = (
                db.query(Conversation)
                .filter_by(session_id=session_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": str(c.id),
                    "session_id": c.session_id,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "message_count": len(c.messages),
                }
                for c in convs
            ]
