"""Recherche sur Internet, utilisée comme outil par l'agent.

On utilise la bibliothèque ddgs (DuckDuckGo et autres moteurs) : gratuite et sans clé API.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearcher(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[WebResult]: ...


class DDGSWebSearcher:
    """Recherche web via ddgs. Renvoie une liste vide en cas de problème réseau."""

    def __init__(self, region: str = "fr-fr"):
        self.region = region

    def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        from ddgs import DDGS  # importé ici : la bibliothèque n'est utile que si la recherche web est activée

        try:
            raw = DDGS().text(query, region=self.region, max_results=max_results)
        except Exception as error:  # réseau coupé, moteur indisponible, limite atteinte...
            raise RuntimeError(f"La recherche web a échoué : {error}") from error
        return [
            WebResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
            for r in raw or []
        ]
