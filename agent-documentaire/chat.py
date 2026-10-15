"""Étape 2 : discuter avec l'agent dans le terminal.

Usage : python chat.py
"""

import openai

from app.factory import make_agent


def main() -> None:
    agent = make_agent()
    print("Agent documentaire prêt. Posez une question (ou 'quit' pour sortir).\n")
    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        try:
            answer = agent.ask(question)
        except openai.RateLimitError:
            print("\nQuota atteint (erreur 429) : la limite gratuite de ce modèle est épuisée pour aujourd'hui.")
            print("Changez CHAT_MODEL dans .env (chaque modèle a son propre quota) ou utilisez un autre fournisseur.\n")
            continue
        except openai.APIStatusError as e:
            print(f"\nErreur du fournisseur ({e.status_code}) : {e.message}")
            if e.status_code == 503:
                print("Le modèle est surchargé : réessayez dans un instant ou changez CHAT_MODEL dans .env.")
            print()
            continue
        except openai.APITimeoutError:
            print("\nLe modèle met trop de temps à répondre (il est sans doute surchargé). Réessayez dans un instant ou changez CHAT_MODEL dans .env.\n")
            continue
        except openai.APIConnectionError:
            print("\nImpossible de joindre le fournisseur. Vérifiez votre connexion et LLM_BASE_URL dans .env.\n")
            continue
        print(f"\nAgent > {answer.text}")
        tokens = answer.usage["prompt_tokens"] + answer.usage["completion_tokens"]
        print(f"        sources : {', '.join(answer.sources) or 'aucune'} | {answer.model} | {answer.latency_ms:.0f} ms | {tokens} tokens\n")


if __name__ == "__main__":
    main()
