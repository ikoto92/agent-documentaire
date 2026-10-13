"""Jeu d'évaluation : mesure la qualité de l'agent et détecte les régressions.

Deux métriques :
- retrieval : la bonne source fait-elle partie des extraits retrouvés ?
- réponse   : la réponse contient-elle les informations attendues ?

Usage :
    python -m evals.run_evals                    # évaluation complète (LLM + recherche)
    python -m evals.run_evals --retrieval-only   # recherche seule, sans appeler le LLM de chat
    python -m evals.run_evals --min-score 0.8    # échoue si le score est inférieur à 80 %
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.factory import make_agent, make_embedder
from app.store import VectorStore

DATASET = Path(__file__).parent / "dataset.json"
RESULTS = Path(__file__).parent / "results.json"


def keywords_ok(text: str, keywords: list[str], match: str = "all") -> bool:
    text = text.lower()
    hits = [k.lower() in text for k in keywords]
    return any(hits) if match == "any" else all(hits)


def retrieval_ok(sources: list[str], expected: str | None) -> bool:
    if expected is None:  # question hors sujet : pas de source attendue
        return True
    return any(s.split("#")[0] == expected for s in sources)


def run(retrieval_only: bool) -> dict:
    settings = get_settings()
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = []

    if retrieval_only:
        store = VectorStore.load(settings.index_path)
        embedder = make_embedder(settings)
        for case in cases:
            results = store.search(embedder.embed([case["question"]])[0], settings.top_k)
            sources = [c.id for c, _ in results]
            rows.append({"question": case["question"], "retrieval_ok": retrieval_ok(sources, case["expected_source"]), "sources": sources})
    else:
        agent = make_agent(settings)
        for case in cases:
            answer = agent.ask(case["question"])
            rows.append(
                {
                    "question": case["question"],
                    "retrieval_ok": retrieval_ok(answer.sources, case["expected_source"]),
                    "answer_ok": keywords_ok(answer.text, case["expected_keywords"], case.get("match", "all")),
                    "answer": answer.text,
                    "sources": answer.sources,
                    "latency_ms": answer.latency_ms,
                    "tokens": answer.usage["prompt_tokens"] + answer.usage["completion_tokens"],
                }
            )

    n = len(rows)
    summary = {"cases": n, "retrieval_score": sum(r["retrieval_ok"] for r in rows) / n}
    if not retrieval_only:
        summary["answer_score"] = sum(r["answer_ok"] for r in rows) / n
        summary["avg_latency_ms"] = round(sum(r["latency_ms"] for r in rows) / n, 1)
        summary["total_tokens"] = sum(r["tokens"] for r in rows)
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    report = run(args.retrieval_only)
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in report["rows"]:
        status = "OK " if row["retrieval_ok"] and row.get("answer_ok", True) else "KO "
        print(f"{status} {row['question']}")
    s = report["summary"]
    print("\n--- Résumé ---")
    for key, value in s.items():
        print(f"{key:>16} : {value:.0%}" if key.endswith("score") else f"{key:>16} : {value}")
    print(f"Détails dans {RESULTS}")

    score = s.get("answer_score", s["retrieval_score"])
    if score < args.min_score:
        print(f"Score {score:.0%} inférieur au minimum demandé ({args.min_score:.0%}) : régression détectée.")
        sys.exit(1)


if __name__ == "__main__":
    main()
