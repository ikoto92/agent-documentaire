"""Assemble les briques (client LLM, embeddings, index, agent) à partir de la configuration."""

from openai import OpenAI

from app.agent import Agent
from app.config import Settings, get_settings
from app.embeddings import OpenAICompatibleEmbedder
from app.store import VectorStore
from app.tracing import Tracer
from app.web import DDGSWebSearcher


def make_client(settings: Settings) -> OpenAI:
    # max_retries=1 et timeout=40 s : si le modèle est surchargé ou trop lent, on n'insiste pas
    # longtemps et l'agent passe vite au modèle de secours (CHAT_MODEL_FALLBACK).
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=1, timeout=40)


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
        web_searcher=DDGSWebSearcher() if settings.web_search else None,
        web_max_results=settings.web_max_results,
        fallback_models=settings.fallback_models,
    )
