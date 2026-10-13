"""Étape 2 : discuter avec l'agent dans le terminal.

Usage : python chat.py
"""

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
        answer = agent.ask(question)
        print(f"\nAgent > {answer.text}")
        tokens = answer.usage["prompt_tokens"] + answer.usage["completion_tokens"]
        print(f"        sources : {', '.join(answer.sources) or 'aucune'} | {answer.latency_ms:.0f} ms | {tokens} tokens\n")


if __name__ == "__main__":
    main()
