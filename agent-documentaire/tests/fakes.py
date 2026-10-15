"""Faux composants pour tester sans appeler de vrai modèle (tests rapides, gratuits et reproductibles)."""

import hashlib
import re
from types import SimpleNamespace

import numpy as np

DIM = 256


class FakeEmbedder:
    """Embedding « sac de mots » : chaque mot active une case du vecteur."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = np.zeros(DIM, dtype=np.float32)
            for word in re.findall(r"\w+", text.lower()):
                if len(word) > 3:
                    v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
            vectors.append(v.tolist())
        return vectors


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeChatClient:
    """Simule un LLM : demande d'abord l'outil search_documents, puis répond."""

    def __init__(self, final_answer: str = "Vous avez 25 jours de congés payés [conges.md#0]."):
        self.final_answer = final_answer
        self.calls: list[list[dict]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, tools=None, temperature=0):
        self.calls.append(list(messages))
        already_searched = any(m["role"] == "tool" for m in messages)
        if not already_searched:
            question = messages[-1]["content"]
            return _response(tool_calls=[_tool_call("call_1", "search_documents", f'{{"query": "{question}"}}')])
        return _response(content=self.final_answer)


class LoopingChatClient(FakeChatClient):
    """Simule un LLM qui ne s'arrête jamais d'appeler des outils."""

    def _create(self, model, messages, tools=None, temperature=0):
        return _response(tool_calls=[_tool_call("call_x", "list_documents", "{}")])


class FakeWebSearcher:
    """Simule un moteur de recherche sans accès réseau."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 5):
        from app.web import WebResult

        self.queries.append(query)
        if self.fail:
            raise RuntimeError("La recherche web a échoué : réseau indisponible")
        return [WebResult(title="ETNA – école d'informatique", url="https://etna.io", snippet="École en alternance à Ivry-sur-Seine.")]


class WebChatClient(FakeChatClient):
    """Simule un LLM qui choisit l'outil search_web, puis répond."""

    def _create(self, model, messages, tools=None, temperature=0):
        self.calls.append(list(messages))
        self.tools_seen = [t["function"]["name"] for t in (tools or [])]
        if not any(m["role"] == "tool" for m in messages):
            return _response(tool_calls=[_tool_call("call_w", "search_web", '{"query": "ETNA école"}')])
        return _response(content="L'ETNA est une école en alternance [https://etna.io].")


class ProviderError(Exception):
    """Imite une erreur HTTP du fournisseur (comme openai.APIStatusError)."""

    def __init__(self, status_code: int):
        super().__init__(f"Erreur {status_code}")
        self.status_code = status_code


class OverloadedClient(FakeChatClient):
    """Le modèle 'principal' est surchargé (503) ; les autres modèles répondent normalement."""

    def __init__(self, status_code: int = 503):
        super().__init__()
        self.status_code = status_code
        self.models_called: list[str] = []

    def _create(self, model, messages, tools=None, temperature=0):
        self.models_called.append(model)
        if model == "principal":
            raise ProviderError(self.status_code)
        return super()._create(model, messages, tools, temperature)
