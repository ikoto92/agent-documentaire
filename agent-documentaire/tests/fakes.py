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
