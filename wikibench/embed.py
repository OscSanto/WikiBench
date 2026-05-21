"""Query embedding for wikibench — thin wrapper around fastembed."""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

_lock   = threading.Lock()
_model  = None
_model_name: str | None = None


def _get_model(model_name: str):
    global _model, _model_name
    with _lock:
        if _model is None or _model_name != model_name:
            from fastembed import TextEmbedding
            _model      = TextEmbedding(model_name=model_name)
            _model_name = model_name
        return _model


def encode(texts: Sequence[str], model_name: str) -> np.ndarray:
    """Embed one or more texts; returns float32 array of shape (N, dim)."""
    model  = _get_model(model_name)
    vecs   = list(model.embed(texts))
    arr    = np.array(vecs, dtype=np.float32)
    norms  = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def encode_one(text: str, model_name: str) -> np.ndarray:
    """Embed a single text; returns float32 array of shape (1, dim)."""
    return encode([text], model_name)
