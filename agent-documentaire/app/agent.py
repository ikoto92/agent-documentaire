"""L'agent : un LLM qui décide lui-même d'appeler des outils (tool calling).

Boucle :
1. On envoie la question au LLM avec la liste des outils disponibles.
2. Si le LLM demande un outil, on l'exécute et on lui renvoie le résultat.
3. On recommence jusqu'à ce qu'il donne sa réponse finale (ou max_steps).
"""

import json
import time
from dataclasses import dataclass, field

from app.embeddings import Embedder
from app.store import VectorStore
from app.tracing import Tracer

SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions à partir des documents internes de l'entreprise.

Règles :
- Avant de répondre à une question sur les documents, utilise l'outil search_documents.
- Réponds uniquement à partir des extraits trouvés. Si l'information n'y figure pas, réponds que tu ne trouves pas cette information dans les documents.
- Cite tes sources entre crochets avec l'identifiant de l'extrait, par exemple [conges.md#0].
- Réponds en français, de façon claire et concise."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Recherche les passages des documents les plus pertinents pour une requête.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ce que l'on cherche, en langage naturel."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "Liste les documents disponibles.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class AgentAnswer:
    text: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})


class Agent:
    def __init__(
        self,
        client,
        model: str,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 4,
        max_steps: int = 5,
        tracer: Tracer | None = None,
    ):
        self.client = client
        self.model = model
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.max_steps = max_steps
        self.tracer = tracer or Tracer(None)

    # ---------- Outils ----------

    def search_documents(self, query: str) -> list[tuple]:
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, self.top_k)

    def _run_tool(self, name: str, args: dict) -> tuple[str, list[str]]:
        """Exécute un outil et renvoie (texte pour le LLM, identifiants des sources)."""
        if name == "search_documents":
            results = self.search_documents(args.get("query", ""))
            if not results:
                return "Aucun extrait trouvé.", []
            text = "\n\n".join(f"[{c.id}] (score {s:.2f})\n{c.text}" for c, s in results)
            return text, [c.id for c, _ in results]
        if name == "list_documents":
            return "\n".join(self.store.sources()) or "Aucun document.", []
        return f"Outil inconnu : {name}", []

    # ---------- Boucle de l'agent ----------

    def ask(self, question: str) -> AgentAnswer:
        start = time.perf_counter()
        answer = AgentAnswer(text="")
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, temperature=0
            )
            self._add_usage(answer, response)
            message = response.choices[0].message

            if not message.tool_calls:
                answer.text = (message.content or "").strip()
                break

            # Le LLM demande un ou plusieurs outils : on renvoie sa demande telle quelle
            # dans l'historique, puis on exécute les outils.
            messages.append(self._assistant_message(message))
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result, sources = self._run_tool(call.function.name, args)
                answer.tool_calls.append({"name": call.function.name, "args": args})
                for s in sources:
                    if s not in answer.sources:
                        answer.sources.append(s)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        else:
            answer.text = "Je n'ai pas réussi à formuler une réponse dans le nombre d'étapes autorisé."

        answer.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        self.tracer.log(
            {
                "question": question,
                "answer": answer.text,
                "sources": answer.sources,
                "tool_calls": answer.tool_calls,
                "latency_ms": answer.latency_ms,
                **answer.usage,
                "model": self.model,
            }
        )
        return answer

    @staticmethod
    def _assistant_message(message) -> dict:
        """Recopie le message du LLM qui demande des outils, sans rien perdre.

        Certains fournisseurs ajoutent des champs à renvoyer obligatoirement au tour suivant.
        Par exemple, les modèles Gemini « qui réfléchissent » joignent à chaque appel d'outil
        une signature de raisonnement (extra_content.google.thought_signature) : si on ne la
        renvoie pas, l'API répond par une erreur 400. On garde donc le message complet.
        """
        if hasattr(message, "model_dump"):
            data = message.model_dump(exclude_none=True)
            data["role"] = "assistant"
            data["content"] = message.content or ""
            return data
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ],
        }

    @staticmethod
    def _add_usage(answer: AgentAnswer, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        answer.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        answer.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
