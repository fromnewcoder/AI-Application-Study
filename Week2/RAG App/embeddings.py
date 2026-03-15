"""
embeddings.py — Local embedding function for ChromaDB.

Primary:  sentence-transformers all-MiniLM-L6-v2  (384-dim, semantic search)
Fallback: sklearn HashingVectorizer                (512-dim, keyword search)

The fallback is used automatically if sentence-transformers / torch are not
installed.  On your own machine, run:
    pip install sentence-transformers
and the semantic model will be picked up on the next import.
"""

from __future__ import annotations

from chromadb import EmbeddingFunction, Documents, Embeddings

# ── Try to load sentence-transformers ────────────────────────────────────────
try:
    # Catch OSError too: partially installed torch (missing .so) raises OSError
    # before we ever reach ImportError.
    from sentence_transformers import SentenceTransformer as _ST
    _MODEL_NAME = "all-MiniLM-L6-v2"
    _st_model: _ST | None = None          # lazy-load on first call

    def _get_st_model() -> _ST:
        global _st_model
        if _st_model is None:
            _st_model = _ST(_MODEL_NAME)
        return _st_model

    EMBEDDING_BACKEND = "sentence-transformers"

    class LocalEmbeddingFunction(EmbeddingFunction):
        """Semantic embeddings via sentence-transformers (all-MiniLM-L6-v2)."""

        def __call__(self, input: Documents) -> Embeddings:
            model = _get_st_model()
            vecs = model.encode(list(input), normalize_embeddings=True)
            return vecs.tolist()

        @property
        def dim(self) -> int:
            return 384

        @property
        def backend(self) -> str:
            return f"sentence-transformers ({_MODEL_NAME})"

except Exception:
    # ── Fallback: sklearn HashingVectorizer ───────────────────────────────────
    # Catches ImportError, OSError (missing .so from partial torch install), etc.
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer as _HV

    EMBEDDING_BACKEND = "sklearn-hashing"
    _N_FEATURES = 512

    class LocalEmbeddingFunction(EmbeddingFunction):          # type: ignore[no-redef]
        """
        Stateless hash-based embeddings (no model download required).

        Works well for keyword-heavy queries.  For true semantic search,
        install sentence-transformers:
            pip install sentence-transformers
        """

        def __init__(self) -> None:
            self._vec = _HV(
                n_features=_N_FEATURES,
                norm="l2",
                alternate_sign=False,
                analyzer="word",
                ngram_range=(1, 2),   # unigrams + bigrams for better recall
            )

        def __call__(self, input: Documents) -> Embeddings:
            matrix = self._vec.transform(list(input)).toarray().astype(float)
            return matrix.tolist()

        @property
        def dim(self) -> int:
            return _N_FEATURES

        @property
        def backend(self) -> str:
            return "sklearn HashingVectorizer (bigrams, 512-dim)"
