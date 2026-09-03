"""
embeddings.py
-------------
Generate sentence embeddings locally using Sentence-Transformers.

These models run entirely on your own machine (CPU is fine) after the
first download from Hugging Face — no API keys and no paid services
are involved.
"""

from typing import List

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache loaded models in-process so repeated calls don't reload from disk.
_model_cache = {}


def load_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load (and cache) a SentenceTransformer model by name."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(
    texts: List[str], model_name: str = DEFAULT_MODEL_NAME, normalize: bool = True
) -> np.ndarray:
    """Embed a list of strings into a (n_texts, embedding_dim) float32 array."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    model = load_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embed_text(
    text: str, model_name: str = DEFAULT_MODEL_NAME, normalize: bool = True
) -> np.ndarray:
    """Embed a single string into a 1D float32 vector."""
    return embed_texts([text], model_name=model_name, normalize=normalize)[0]
