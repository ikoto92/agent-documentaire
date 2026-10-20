"""Transformation de textes en vecteurs (embeddings)."""

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    """Embeddings via n'importe quelle API compatible OpenAI (Ollama, Gemini, OpenAI...)."""

    def __init__(self, client, model: str, batch_size: int = 64):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
