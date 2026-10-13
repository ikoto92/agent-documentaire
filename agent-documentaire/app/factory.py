"""Assemble les briques (client LLM, embeddings, index, agent) à partir de la configuration."""

from openai import OpenAI

from app.agent import Agent
from app.config import Settings, get_settings
from app.embeddings import OpenAICompatibleEmbedder
from app.store import VectorStore
from app.tracing import Tracer


def make_client(settings: Settings) -> OpenAI:
    # max_retries : en cas de surcharge (erreur 503), le client réessaie automatiquement
    # en attendant un peu plus à chaque fois. On reste à 2 pour ne pas épuiser le quota gratuit.
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=2, timeout=60)


def make_embedder(settings: Settings) -> OpenAICompatibleEmbedder:
    client = OpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key, max_retries=2, timeout=60)
    return OpenAICompatibleEmbedder(client, settings.embed_model)


def make_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()
    return Agent(
        client=make_client(settings),
        model=settings.chat_model,
        store=VectorStore.load(settings.index_path),
        embedder=make_embedder(settings),
        top_k=settings.top_k,
        tracer=Tracer(settings.trace_path),
    )
