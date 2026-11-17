"""Defaults and presets."""
import os


def _normalize_host(h):
    h = (h or "").strip() or "http://localhost:11434"
    if not h.startswith(("http://", "https://")):
        h = "http://" + h
    h = h.replace("0.0.0.0", "localhost")
    if ":" not in h.split("://", 1)[1]:
        h += ":11434"
    return h.rstrip("/")


# Ollama itself reads OLLAMA_HOST, so we honour it too.
DEFAULT_HOST = _normalize_host(os.environ.get("OLLAMA_HOST"))
DEFAULT_MODEL = os.environ.get("XAI_OLLAMA_MODEL", "qwen3.5:4b")

# Attribution cost grows with the number of words, so cap it.
MAX_UNITS = 60

# How many model calls to spend. "samples" = random word-removal experiments,
# "random" = random-removal controls for the faithfulness test.
PRESETS = {
    "Quick": dict(n_samples=120, n_random=39, n_noise=3, n_boot=300),
    "Standard": dict(n_samples=300, n_random=60, n_noise=5, n_boot=500),
    "Thorough": dict(n_samples=800, n_random=120, n_noise=8, n_boot=1000),
}
