"""Ollama streaming client for wikibench.

Measures TTFT client-side (time from POST → first non-empty content token).
Converts Ollama nanosecond fields to milliseconds.
Detects cold-start when load_duration > 2s.
"""
from __future__ import annotations

import json
import time
from typing import Generator

import requests

_DEFAULT_URL = "http://localhost:11434"


def _ns_to_ms(ns: int | None) -> float | None:
    return ns / 1_000_000 if ns is not None else None


def chat(
    model:    str,
    messages: list[dict],
    base_url: str = _DEFAULT_URL,
    timeout:  int = 120,
) -> dict:
    """Stream a chat request; return combined result dict with all metrics.

    Returns:
        {
          "content":               str,   # full assistant reply
          "ttft_ms":               float, # client-measured, ms
          "total_duration_ms":     float | None,
          "load_duration_ms":      float | None,
          "prompt_eval_count":     int   | None,
          "prompt_eval_duration_ms": float | None,
          "eval_count":            int   | None,
          "eval_duration_ms":      float | None,
          "tokens_per_sec":        float | None,
          "prefill_rate":          float | None,
          "cold_start":            bool,
        }
    """
    url     = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}

    t_start  = time.perf_counter()
    ttft_ms  = None
    content  = []
    last_meta: dict = {}

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            token = chunk.get("message", {}).get("content", "")
            if token and ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000

            content.append(token)

            if chunk.get("done"):
                last_meta = chunk
                break

    full_content = "".join(content)

    # Derived metrics
    eval_count    = last_meta.get("eval_count")
    eval_dur_ns   = last_meta.get("eval_duration")
    prompt_count  = last_meta.get("prompt_eval_count")
    prompt_dur_ns = last_meta.get("prompt_eval_duration")
    load_dur_ns   = last_meta.get("load_duration")

    eval_dur_ms   = _ns_to_ms(eval_dur_ns)
    prompt_dur_ms = _ns_to_ms(prompt_dur_ns)
    load_dur_ms   = _ns_to_ms(load_dur_ns)
    total_dur_ms  = _ns_to_ms(last_meta.get("total_duration"))

    tokens_per_sec = (
        eval_count / (eval_dur_ns / 1e9)
        if eval_count and eval_dur_ns and eval_dur_ns > 0
        else None
    )
    prefill_rate = (
        prompt_count / (prompt_dur_ns / 1e9)
        if prompt_count and prompt_dur_ns and prompt_dur_ns > 0
        else None
    )
    cold_start = bool(load_dur_ns and load_dur_ns > 2_000_000_000)

    return {
        "content":                full_content,
        "ttft_ms":                ttft_ms,
        "total_duration_ms":      total_dur_ms,
        "load_duration_ms":       load_dur_ms,
        "prompt_eval_count":      prompt_count,
        "prompt_eval_duration_ms": prompt_dur_ms,
        "eval_count":             eval_count,
        "eval_duration_ms":       eval_dur_ms,
        "tokens_per_sec":         tokens_per_sec,
        "prefill_rate":           prefill_rate,
        "cold_start":             cold_start,
    }


def list_models(base_url: str = _DEFAULT_URL) -> list[str]:
    resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def warmup(model: str, base_url: str = _DEFAULT_URL) -> None:
    """Send a trivial prompt to ensure the model is loaded before benchmarking."""
    chat(model, [{"role": "user", "content": "Hi"}], base_url=base_url)
