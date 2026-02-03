"""
Streamlit web interface for AutomotiveGPT.

Features:
  - Sidebar: vehicle filter dropdowns (make, model, year, subsystem)
  - Chat area: message bubbles with live spinner during generation
  - Source cards: expandable citation blocks below each answer
  - Conversation controls: new / clear conversation
  - Dark theme with custom CSS

The UI calls the FastAPI backend over HTTP so it can run independently.
"""

import uuid
import logging

import streamlit as st
import requests

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

DEFAULT_MAKES = ["All", "Honda", "Toyota", "Ford", "Chevrolet", "BMW",
                 "Mercedes-Benz", "Tesla", "Hyundai", "Nissan",
                 "Volkswagen", "Audi", "Lexus", "Jeep"]
DEFAULT_SUBSYSTEMS = ["All", "Engine", "Transmission", "Brake", "Electrical",
                      "Steering", "Suspension", "Fuel", "Exhaust", "HVAC",
                      "Battery", "Charging", "Cooling"]


# ── Page config & CSS ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AutomotiveGPT",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; }
    .main .block-container { padding-top: 1rem; max-width: 900px; }
    .chat-user {
        background: #1e3a5f; border-radius: 12px 12px 4px 12px;
        padding: 12px 16px; margin: 8px 0 8px auto; max-width: 85%;
        text-align: right; color: #e2e8f0;
    }
    .chat-assistant {
        background: #1a1d2e; border: 1px solid #2a2d3e;
        border-radius: 12px 12px 12px 4px; padding: 12px 16px;
        margin: 8px 0; max-width: 90%; color: #e2e8f0; line-height: 1.6;
    }
    .source-card {
        background: #161829; border: 1px solid #2a2d3e; border-radius: 8px;
        padding: 10px 14px; margin: 4px 0; font-size: 0.85rem; color: #94a3b8;
    }
    .source-card .src-hdr { color: #60a5fa; font-weight: 600; margin-bottom: 4px; }
    .conf-badge {
        display: inline-block; background: #1e3a5f; color: #60a5fa;
        border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
def _init():
    for k, v in {
        "conversation_id": None,
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "filters": {},
    }.items():
        st.session_state.setdefault(k, v)

_init()


# ── API helpers ───────────────────────────────────────────────────────
def _api_chat(message: str) -> dict:
    filters = {k: v for k, v in st.session_state["filters"].items() if v and v != "All"} or None
    payload = {
        "message": message,
        "session_id": st.session_state["session_id"],
        "filters": filters,
    }
    if st.session_state["conversation_id"]:
        payload["conversation_id"] = st.session_state["conversation_id"]
    resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _api_delete(conv_id: str) -> bool:
    return requests.delete(f"{API_BASE}/conversations/{conv_id}", timeout=10).status_code == 200


def _api_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception:
        return {"status": "unreachable", "redis": False, "db": False}


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 AutomotiveGPT")
    st.caption("Technical QA over vehicle service manuals")
    st.divider()

    st.markdown("### 🔍 Vehicle Filters")
    make = st.selectbox("Make", DEFAULT_MAKES, key="sb_make")
    subsys = st.selectbox("Subsystem", DEFAULT_SUBSYSTEMS, key="sb_subsys")
    y1, y2 = st.columns(2)
    year_min = y1.number_input("Year (from)", 1990, 2025, 2020, key="sb_ymin")
    year_max = y2.number_input("Year (to)", 1990, 2025, 2025, key="sb_ymax")

    st.session_state["filters"] = {
        "make": make if make != "All" else None,
        "subsystem": subsys.lower() if subsys != "All" else None,
    }

    st.divider()
    st.markdown("### 💬 Conversation")
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.update(conversation_id=None, messages=[])
        st.rerun()

    if st.session_state["conversation_id"]:
        st.caption(f"ID: `{st.session_state['conversation_id'][:16]}…`")
        if st.button("🗑️ Clear", use_container_width=True):
            _api_delete(st.session_state["conversation_id"])
            st.session_state.update(conversation_id=None, messages=[])
            st.rerun()

    st.divider()
    st.markdown("### ⚡ Status")
    h = _api_health()
    st.markdown(
        f"API: {'🟢' if h['status'] == 'ok' else '🔴'} | "
        f"Redis: {'🟢' if h.get('redis') else '🔴'} | "
        f"DB: {'🟢' if h.get('db') else '🔴'}"
    )


# ── Chat area ─────────────────────────────────────────────────────────
st.markdown("## 💬 Ask a Question")
st.caption("Ask anything about vehicle service, repair procedures, or specs.")

# Render history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        conf = msg.get("confidence", 0)
        badge = f'<span class="conf-badge">Confidence: {conf:.0%}</span>'
        st.markdown(
            f'<div class="chat-assistant">{msg["content"]} {badge}</div>',
            unsafe_allow_html=True,
        )
        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
                for s in sources:
                    pg = f"Page {s['page']}" if s.get("page") else "—"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="src-hdr">[Source {s["source_id"]}] {s["source_file"]}</div>'
                        f'{pg} · {s.get("section_type","—")} · score {s.get("score",0):.2f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about a vehicle service procedure…")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.spinner("🔍 Searching manuals and generating answer…"):
        try:
            result = _api_chat(user_input)
            st.session_state["conversation_id"] = result["conversation_id"]
            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "confidence": result.get("confidence", 0),
                "sources": result.get("sources", []),
            })
            cached = " (cached)" if result.get("cached") else ""
            st.caption(f"⏱️ {result.get('latency_ms', 0)} ms{cached}")
        except requests.exceptions.ConnectionError:
            st.error("❌ API server unreachable. Start FastAPI on port 8000.")
            st.session_state["messages"].pop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.session_state["messages"].pop()

    st.rerun()
