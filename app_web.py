"""Interface web de l'agent documentaire (Streamlit).

Usage en local : streamlit run app_web.py
En ligne : Streamlit Community Cloud ou Hugging Face Spaces (voir le README).
"""

import os

import openai
import streamlit as st

st.set_page_config(page_title="Agent documentaire IA", page_icon="📚", layout="centered")

# En ligne, la clé API est rangée dans les « Secrets » de l'hébergeur, jamais dans le code.
# On les recopie dans les variables d'environnement avant de lire la configuration.
SECRETS_PROBLEM = ""
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ[key] = str(value).strip()
except FileNotFoundError:  # pas de fichier de secrets en local : on utilise le .env
    pass
except Exception as error:  # secrets mal écrits (guillemets, ligne invalide...)
    SECRETS_PROBLEM = f"Les Secrets sont mal écrits et ont été ignorés : {error}"

from app.config import get_settings  # noqa: E402  (après la lecture des secrets)
from app.display import format_answer_markdown  # noqa: E402
from app.factory import make_agent  # noqa: E402
from ingest import build_index  # noqa: E402

# Limite de questions par visiteur, pour protéger le quota gratuit de la clé API.
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "5"))

EXAMPLES = [
    "Combien de jours de télétravail par semaine sont autorisés ?",
    "Quel est le plafond de remboursement d'un repas ?",
    "Qu'est-ce que le RAG en intelligence artificielle ?",
]


@st.cache_resource(show_spinner="Préparation de l'agent (indexation des documents)…")
def load_agent():
    settings = get_settings()
    if not os.path.exists(settings.index_path):
        if build_index(settings) is None:
            raise RuntimeError(f"Aucun document trouvé dans {settings.docs_dir}.")
    return make_agent(settings)


def show_answer(entry: dict) -> None:
    st.markdown(entry["text"])
    if entry["sources"]:
        label = "Sources" if entry["cited"] else "Extraits consultés"
        lines = []
        for i, source in enumerate(entry["sources"], 1):
            link = f"[{source}]({source})" if source.startswith("http") else f"`{source}`"
            lines.append(f"{i}. {link}")
        with st.expander(label, expanded=entry["cited"]):
            st.markdown("\n".join(lines))
    st.caption(entry["meta"])


def friendly_error(error: Exception) -> str:
    if isinstance(error, openai.RateLimitError):
        return "Le quota gratuit du modèle est épuisé pour aujourd'hui. Réessayez demain."
    if isinstance(error, openai.APITimeoutError):
        return "Le modèle met trop de temps à répondre. Réessayez dans un instant."
    if isinstance(error, openai.APIStatusError) and error.status_code == 503:
        return "Le modèle est surchargé. Réessayez dans un instant."
    if isinstance(error, openai.APIConnectionError):
        return "Impossible de joindre le fournisseur du modèle."
    return "Une erreur est survenue. Réessayez dans un instant."


# ---------------- Page ----------------
st.title("📚 Agent documentaire IA")
st.write(
    "Un agent qui répond à partir de documents internes (une entreprise **fictive**, NovaTech : "
    "congés, télétravail, notes de frais) et, si besoin, du web, **en citant ses sources**."
)
with st.expander("Comment ça marche ?"):
    st.markdown(
        "- **RAG** : les documents sont découpés en morceaux, transformés en vecteurs, et l'agent "
        "retrouve les passages les plus proches de la question.\n"
        "- **Tool calling** : le modèle choisit lui-même ses outils (recherche dans les documents, "
        "liste des documents, recherche web).\n"
        "- **Fiabilité** : jeu d'évaluation, tests automatisés, modèle de secours en cas de panne, "
        "et suivi de la latence et des tokens.\n\n"
        "Code source : [github.com/ikoto92](https://github.com/ikoto92) – André IKOTO"
    )

try:
    agent = load_agent()
except Exception as error:  # clé absente, documents manquants...
    settings = get_settings()
    st.error(f"L'agent n'a pas pu démarrer : {error}")
    diagnostic = [
        f"- Adresse du modèle (LLM_BASE_URL) : `{settings.base_url}`",
        f"- Adresse des embeddings : `{settings.embed_base_url}`",
        f"- Clé API (LLM_API_KEY) : {'présente' if settings.api_key not in ('', 'ollama') else '**absente**'}",
        f"- Modèles : `{settings.chat_model}` / embeddings `{settings.embed_model}`",
    ]
    if SECRETS_PROBLEM:
        diagnostic.insert(0, f"- {SECRETS_PROBLEM}")
    if "localhost" in settings.base_url:
        diagnostic.append("- L'agent cherche un modèle local (Ollama) : en ligne, renseignez les **Secrets** de l'application.")
    st.markdown("**Diagnostic** (la clé n'est jamais affichée) :\n" + "\n".join(diagnostic))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        show_answer(entry)

remaining = MAX_QUESTIONS - len(st.session_state.history)

question = None
if not st.session_state.history:
    st.caption("Exemples :")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}"):
            question = example

if remaining <= 0:
    st.info(f"Vous avez atteint la limite de {MAX_QUESTIONS} questions pour cette démonstration. Merci de votre visite !")
else:
    typed = st.chat_input(f"Posez votre question ({remaining} restante{'s' if remaining > 1 else ''})")
    question = typed or question

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("L'agent cherche…"):
            try:
                answer = agent.ask(question)
            except Exception as error:
                st.error(friendly_error(error))
                st.stop()
        text, sources, cited = format_answer_markdown(answer.text, answer.sources)
        tokens = answer.usage["prompt_tokens"] + answer.usage["completion_tokens"]
        entry = {
            "question": question,
            "text": text,
            "sources": sources,
            "cited": cited,
            "meta": f"{answer.model} · {answer.latency_ms / 1000:.1f} s · {tokens} tokens",
        }
        show_answer(entry)
    st.session_state.history.append(entry)
    st.rerun()
