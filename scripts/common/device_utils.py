#!/usr/bin/env python3
"""Portable inference-device resolution.

Every script previously hardcoded --device default="mps", which only works
on Apple Silicon and breaks on CUDA/Windows/Linux/CPU-only machines. This
picks the best available device at runtime instead, with an optional DEVICE
env var / --device flag to force a specific one.
"""

import os


def resolve_device(requested: str | None = None) -> str:
    """Returns `requested` (or the DEVICE env var) if set, otherwise
    auto-detects: CUDA > MPS > CPU."""
    device = requested or os.environ.get("DEVICE")
    if device:
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
