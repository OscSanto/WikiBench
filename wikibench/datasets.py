"""MedQA dataset loader and prompt builder for wikibench."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


# ── MedQA record ──────────────────────────────────────────────────────────────

def load_medqa(path: str | Path, n: int | None = None) -> list[dict]:
    """Load MedQA JSONL (US 4- or 5-option format).

    Each returned dict has:
      question_idx  : int
      question_text : str
      options       : {"A": "...", "B": "...", ...}
      answer        : "A" | "B" | ... (correct letter)
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            options = _normalise_options(raw)
            answer  = _normalise_answer(raw, options)
            records.append({
                "question_idx":  i,
                "question_text": raw["question"],
                "options":       options,
                "answer":        answer,
            })
    return records


def _normalise_options(raw: dict) -> dict[str, str]:
    """Convert various MedQA option formats to {"A": text, ...}."""
    opts = raw.get("options") or raw.get("choices") or {}
    if isinstance(opts, dict):
        return {k.upper(): v for k, v in opts.items()}
    if isinstance(opts, list):
        letters = "ABCDE"
        return {letters[i]: str(v) for i, v in enumerate(opts) if i < len(letters)}
    return {}


def _normalise_answer(raw: dict, options: dict[str, str]) -> str:
    """Return the correct answer letter (uppercase A–E)."""
    ans = raw.get("answer") or raw.get("answer_idx") or raw.get("correct") or ""
    ans = str(ans).strip().upper()

    # Already a letter
    if ans in options:
        return ans

    # Might be a full option text — match against options
    ans_lower = ans.lower()
    for letter, text in options.items():
        if text.strip().lower() == ans_lower:
            return letter

    # First character if it's a letter
    if ans and ans[0] in options:
        return ans[0]

    return ans  # return as-is; pipeline will treat as parse failure


# ── Prompt building ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a medical knowledge assistant. "
    "Answer the multiple-choice question using ONLY the provided context. "
    "Respond with exactly one letter (A, B, C, D, or E) on the first line, "
    "optionally followed by a brief explanation."
)


def format_prompt(
    question_text: str,
    options:       dict[str, str],
    context_chunks: list[dict],
) -> list[dict]:
    """Build the Ollama messages list for a MedQA question.

    Returns [{"role": "system", ...}, {"role": "user", ...}].
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        title   = chunk.get("title", "")
        section = chunk.get("section", "")
        text    = chunk.get("text", "")
        header  = f"[{i}] {title}" + (f" › {section}" if section else "")
        context_parts.append(f"{header}\n{text}")

    context_block = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"

    options_block = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    user_msg = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question_text}\n\n"
        f"{options_block}\n\n"
        "Answer (letter only on first line):"
    )

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


# ── Answer parsing ────────────────────────────────────────────────────────────

_LETTER_RE  = re.compile(r"\b([A-E])\b", re.IGNORECASE)
_PREFIX_RE  = re.compile(r"^\s*([A-E])[.):]\s", re.IGNORECASE)
_ALONE_RE   = re.compile(r"^\s*([A-E])\s*$", re.IGNORECASE)


def parse_answer(raw_response: str, valid_letters: set[str] | None = None) -> str | None:
    """Extract the predicted answer letter from the model's raw response.

    Cascade:
      1. Standalone letter on line 1
      2. Regex for «A.» / «A)» / «A:» prefix on line 1
      3. First letter found anywhere in the response
    Returns None if no letter found.
    """
    valid = valid_letters or {"A", "B", "C", "D", "E"}
    first_line = raw_response.strip().splitlines()[0] if raw_response.strip() else ""

    m = _ALONE_RE.match(first_line)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    m = _PREFIX_RE.match(first_line)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    for m in _LETTER_RE.finditer(raw_response):
        if m.group(1).upper() in valid:
            return m.group(1).upper()

    return None
