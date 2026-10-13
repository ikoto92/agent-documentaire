"""Configuration lue depuis le fichier .env (voir .env.example)."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")  # utf-8-sig : accepte aussi les fichiers enregistrés « avec BOM »


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Fournisseur compatible avec l'API OpenAI (Ollama, Gemini, OpenAI, Mistral...)
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    # Facultatif : un autre fournisseur pour les embeddings (sinon, le même que pour le chat)
    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL", _env("LLM_BASE_URL", "http://localhost:11434/v1")))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY", _env("LLM_API_KEY", "ollama")))

    # Fichiers
    docs_dir: str = field(default_factory=lambda: _env("DOCS_DIR", "data/docs"))
    index_path: str = field(default_factory=lambda: _env("INDEX_PATH", "data/index.json"))
    trace_path: str = field(default_factory=lambda: _env("TRACE_PATH", "data/traces.jsonl"))

    # Découpage et recherche
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))


def get_settings() -> Settings:
    return Settings()
