"""Petite base vectorielle en mémoire, sauvegardée en JSON.

Volontairement simple : on calcule la similarité cosinus avec NumPy.
En production on utiliserait une vraie base (pgvector, Qdrant, Azure AI Search...).
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.loader import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Il faut autant de vecteurs que de chunks")
        if not chunks:
            return
        new = self._normalize(np.asarray(vectors, dtype=np.float32))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        self.chunks.extend(chunks)

    def search(self, query_vector: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        """Renvoie les k chunks les plus proches de la requête, avec leur score (0 à 1)."""
        if self._matrix is None or not self.chunks:
            return []
        query = self._normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._matrix @ query  # similarité cosinus (vecteurs normalisés)
        best = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in best]

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": [] if self._matrix is None else self._matrix.tolist(),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        if not Path(path).exists():
            raise FileNotFoundError(f"Index introuvable : {path}. Lancez d'abord : python ingest.py")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        store.add([Chunk(**c) for c in data["chunks"]], data["vectors"])
        return store

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms
