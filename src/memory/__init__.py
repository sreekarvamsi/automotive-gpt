"""
Conversation memory store (PostgreSQL via SQLAlchemy).

Schema
------
conversations
  id          TEXT  PK   – caller-supplied or auto-generated UUID
  created_at  TIMESTAMP
  updated_at  TIMESTAMP
  metadata    JSONB      – optional bag (e.g. active filters, user agent)

messages
  id              SERIAL PK
  conversation_id TEXT  FK → conversations.id
  role            TEXT       – "user" | "assistant"
  content         TEXT       – the message body
  retrieval_ctx   JSONB      – the context chunks that backed this reply (assistant only)
  timing          JSONB      – latency metadata (assistant only)
  created_at      TIMESTAMP
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column, Text, Integer, DateTime, ForeignKey, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.config import settings

logger = logging.getLogger("automotive_gpt.conversation_store")

Base = declarative_base()
engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


# ─────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )
    metadata_ = Column("metadata", JSON, default=dict)

    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Text, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(Text, nullable=False)            # "user" | "assistant"
    content = Column(Text, nullable=False)
    retrieval_ctx = Column(JSON, nullable=True)    # list of retrieval result dicts
    timing = Column(JSON, nullable=True)           # {"first_token_ms": …, "total_ms": …}
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


# ─────────────────────────────────────────────────────────────
# Database initialisation
# ─────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist.  Idempotent."""
    Base.metadata.create_all(engine)
    logger.info("Database tables initialised.")


# ─────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────

class ConversationStore:
    """All database operations for conversations and messages."""

    def __init__(self):
        self.SessionLocal = SessionLocal

    # ── Conversations ─────────────────────────────────────────

    def get_or_create_conversation(
        self,
        conversation_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Return an existing conversation ID or create a new one.
        If conversation_id is None, a new UUID is generated.
        """
        with self.SessionLocal() as session:
            if conversation_id:
                conv = session.get(Conversation, conversation_id)
                if conv:
                    return conv.id

            new_id = conversation_id or str(uuid.uuid4())
            conv = Conversation(id=new_id, metadata_=metadata or {})
            session.add(conv)
            session.commit()
            logger.debug("Created conversation: %s", new_id)
            return new_id

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages. Returns True if found."""
        with self.SessionLocal() as session:
            conv = session.get(Conversation, conversation_id)
            if not conv:
                return False
            session.delete(conv)
            session.commit()
            logger.info("Deleted conversation: %s", conversation_id)
            return True

    # ── Messages ──────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        retrieval_ctx: list[dict] | None = None,
        timing: dict | None = None,
    ) -> int:
        """Append a message to a conversation. Returns the new message ID."""
        with self.SessionLocal() as session:
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                retrieval_ctx=retrieval_ctx,
                timing=timing,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg.id

    def get_conversation_history(
        self,
        conversation_id: str,
        max_turns: int = 20,
    ) -> list[dict[str, str]]:
        """
        Return the last N messages as [{role, content}] dicts,
        ready to pass directly into the OpenAI messages array.
        """
        with self.SessionLocal() as session:
            messages = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(max_turns)
                .all()
            )
            messages.reverse()  # chronological order
            return [{"role": m.role, "content": m.content} for m in messages]

    def get_full_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """
        Return the full conversation including all messages with
        retrieval context and timing metadata.
        """
        with self.SessionLocal() as session:
            conv = session.get(Conversation, conversation_id)
            if not conv:
                return None
            return {
                "id": conv.id,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                "metadata": conv.metadata_,
                "messages": [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "retrieval_ctx": m.retrieval_ctx,
                        "timing": m.timing,
                        "created_at": m.created_at.isoformat() if m.created_at else None,
                    }
                    for m in conv.messages
                ],
            }
