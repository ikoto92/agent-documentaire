"""Étape 1 : lit les documents, calcule les embeddings et sauvegarde l'index.

Usage : python ingest.py
"""

import time

from app.config import get_settings
from app.factory import make_embedder
from app.loader import load_documents
from app.store import VectorStore


def main() -> None:
    settings = get_settings()
    start = time.perf_counter()

    chunks = load_documents(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print(f"Aucun document trouvé dans {settings.docs_dir} (formats acceptés : .pdf, .md, .txt)")
        return

    vectors = make_embedder(settings).embed([c.text for c in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    store.save(settings.index_path)

    print(f"{len(store.sources())} documents, {len(chunks)} chunks indexés en {time.perf_counter() - start:.1f} s")
    print(f"Index sauvegardé dans {settings.index_path}")


if __name__ == "__main__":
    main()
