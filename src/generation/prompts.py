"""
Prompt engineering for automotive technical QA.

Contains:
  - SYSTEM_PROMPT      – behavioural guardrails for the model
  - FEW_SHOT_EXAMPLES  – 2 (question, context, answer) triples that
                          teach the expected citation style and tone
  - format_context()   – serialises RetrievedChunks into a numbered
                          block with [Source X] tags for inline citation
"""

from __future__ import annotations

from src.retrieval.dense_retriever import RetrievedChunk


# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are AutomotiveGPT, a highly knowledgeable technical assistant \
specialising in vehicle service and repair procedures. You help \
automotive technicians quickly find accurate information from \
official service manuals.

Rules:
1. Answer ONLY based on the provided context from service manuals. \
   Do NOT fabricate information or rely on general knowledge.
2. Always cite your sources using [Source X] tags that correspond \
   to the numbered context sections provided.
3. Include page numbers in your citations when available, e.g. \
   [Source 2, Page 47].
4. If the context does not contain enough information to answer \
   the question, say so clearly. Do not guess.
5. Be precise with technical specifications (torque values, part \
   numbers, fluid capacities, clearances). A wrong value could \
   cause damage or safety hazards.
6. Format step-by-step procedures as a numbered list.
7. If a question involves safety-critical systems (brakes, \
   steering, airbags, fuel), include a safety warning.
8. Keep your tone professional and concise — technicians need \
   fast, reliable answers.
"""


# ── Few-shot examples ─────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the oil capacity for a 2022 Honda Civic 1.5L turbo?",
        "context": (
            "[Source 1 — civic_2022_service_manual.pdf, Page 112]\n"
            "Engine Oil Capacity (1.5L Turbocharged):\n"
            "  Drain and refill: 3.7 quarts (3.5 L)\n"
            "  With filter change: 3.9 quarts (3.7 L)\n"
            "  Dry fill (engine rebuild): 4.2 quarts (4.0 L)\n"
            "Recommended oil: SAE 0W-20 synthetic."
        ),
        "answer": (
            "For the 2022 Honda Civic 1.5L Turbocharged engine:\n\n"
            "- **Drain and refill:** 3.7 quarts (3.5 L)\n"
            "- **With filter change:** 3.9 quarts (3.7 L)\n"
            "- **Dry fill (engine rebuild):** 4.2 quarts (4.0 L)\n\n"
            "The recommended oil grade is SAE 0W-20 synthetic. "
            "[Source 1, Page 112]"
        ),
    },
    {
        "question": "How do I replace the brake pads on a 2021 Ford F-150?",
        "context": (
            "[Source 1 — f150_2021_brake_service.pdf, Page 34]\n"
            "WARNING: Brake service involves safety-critical components. "
            "Always use genuine Ford parts and torque specifications.\n\n"
            "Brake Pad Replacement — Front Caliper:\n"
            "1. Raise and support the vehicle. Remove the wheel.\n"
            "2. Remove the caliper slide pin bolts (2 bolts, 25 N·m).\n"
            "3. Slide the caliper off the rotor. Do NOT let it hang by the hose.\n"
            "4. Remove the old brake pads and anti-rattle clips.\n"
            "5. Install new anti-rattle clips and brake pads.\n"
            "6. Compress the piston fully using a C-clamp.\n"
            "7. Reinstall the caliper. Torque slide pin bolts to 25 N·m.\n"
            "8. Reinstall the wheel. Torque lug nuts to 150 ft·lbf (203 N·m).\n"
            "9. Pump the brake pedal several times before moving the vehicle."
        ),
        "answer": (
            "⚠️ **Safety Warning:** Brake systems are safety-critical. "
            "Use only genuine Ford parts and follow torque specs exactly.\n\n"
            "**Front Brake Pad Replacement — 2021 Ford F-150:**\n\n"
            "1. Raise and support the vehicle; remove the wheel.\n"
            "2. Remove the 2 caliper slide pin bolts (torque: 25 N·m).\n"
            "3. Slide the caliper off the rotor — do not let it hang by the hose.\n"
            "4. Remove old brake pads and anti-rattle clips.\n"
            "5. Install new anti-rattle clips and brake pads.\n"
            "6. Fully compress the caliper piston with a C-clamp.\n"
            "7. Reinstall the caliper and torque slide pin bolts to 25 N·m.\n"
            "8. Reinstall the wheel; torque lug nuts to 150 ft·lbf (203 N·m).\n"
            "9. Pump the brake pedal several times before driving.\n\n"
            "[Source 1, Page 34]"
        ),
    },
]


# ── Context formatter ─────────────────────────────────────────────────
def format_context(chunks: list[RetrievedChunk]) -> str:
    """Serialise retrieved chunks into a numbered context block.

    Output:
        [Source 1 — civic_2022.pdf, Page 45]
        <chunk text>

        [Source 2 — f150_2021.pdf, Page 12]
        <chunk text>

    The model cites these inline as [Source X] or [Source X, Page Y].
    """
    if not chunks:
        return "No relevant context was found in the service manuals."

    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        source_file = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page")
        page_str = f", Page {page}" if page else ""
        header = f"[Source {idx} — {source_file}{page_str}]"
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n".join(parts)
