"""
Sentence-Transformer embedding wrapper with local caching.
SSL verification is disabled to support corporate/restricted network environments.
"""

import os
import ssl
import logging
import warnings
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _patch_ssl():
    """Disable SSL verification for HuggingFace Hub downloads."""
    try:
        import httpx
        _orig = httpx.Client.__init__
        def _no_verify(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig(self, *args, **kwargs)
        httpx.Client.__init__ = _no_verify

        _orig_async = httpx.AsyncClient.__init__
        def _no_verify_async(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_async(self, *args, **kwargs)
        httpx.AsyncClient.__init__ = _no_verify_async
    except Exception:
        pass

    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass


_patch_ssl()


class EmbeddingModel:
    """Wraps a SentenceTransformer model and provides encode helpers."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join("data", "embeddings")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            warnings.filterwarnings("ignore")
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")

            # Try local cache first, then download if needed
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self.cache_dir,
                device="cpu",
            )
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string and return a flat list of floats."""
        self._load()
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Embed a list of strings in batches."""
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "cache_dir": self.cache_dir,
        }

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
