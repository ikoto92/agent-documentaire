# Agent documentaire (RAG + tool calling)

Un agent IA en Python qui répond aux questions sur des documents internes **en citant ses sources**.
Il cherche lui-même l'information grâce au *tool calling*, s'appuie sur une recherche par similarité (RAG),
et chaque réponse est tracée (latence, tokens) et évaluée sur un jeu de questions.

Les documents d'exemple (`data/docs/`) décrivent une entreprise **fictive**, NovaTech : congés, télétravail, notes de frais.
Vous pouvez les remplacer par vos propres PDF, fichiers Markdown ou texte.

Exemple de session (la réponse exacte dépend du modèle utilisé) :

```
Vous > Combien de jours de télétravail par semaine sont autorisés ?
Agent > Vous pouvez télétravailler jusqu'à 2 jours par semaine, fixés avec votre manager [teletravail.md#0].
        sources : teletravail.md#0, teletravail.md#1 | 1840 ms | 912 tokens
```

## Fonctionnement

```
                      ingest.py (une fois)
  data/docs/*.pdf|md  ──► découpage en chunks ──► embeddings ──► data/index.json
                                                                       │
  chat.py                                                              ▼
  Question ──► Agent (LLM) ──► appelle l'outil search_documents ──► recherche cosinus
                    ▲                                                  │
                    └──────────── extraits + identifiants ◄────────────┘
                    │
                    ▼
  Réponse avec sources ──► trace JSONL (latence, tokens, outils appelés)
```

| Fichier | Rôle |
|---|---|
| `app/loader.py` | Lecture des PDF / Markdown / texte et découpage en chunks avec chevauchement |
| `app/embeddings.py` | Transformation des textes en vecteurs via une API compatible OpenAI |
| `app/store.py` | Base vectorielle simple (NumPy) : similarité cosinus, sauvegarde JSON |
| `app/agent.py` | Boucle de l'agent : le LLM choisit ses outils, on les exécute, il répond |
| `app/tracing.py` | Observabilité : une ligne JSON par question (latence, tokens, sources) |
| `evals/` | Jeu d'évaluation et script de mesure de la qualité (détection de régressions) |
| `mcp_server.py` | Expose la recherche comme serveur **MCP** utilisable par Claude Desktop, un IDE... |
| `tests/` | Tests automatisés (pytest) avec un faux LLM : rapides, gratuits, reproductibles |
| `.github/workflows/ci.yml` | CI GitHub Actions : les tests tournent à chaque push |

Le code fonctionne avec **n'importe quel fournisseur compatible avec l'API OpenAI** : Ollama (local et gratuit),
Google Gemini, OpenAI, Mistral, Azure OpenAI... Il suffit de changer le fichier `.env`.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows : copy .env.example .env
```

Choisissez ensuite un fournisseur dans `.env` :

**Option 1 – Ollama (gratuit, local).** Installez Ollama depuis https://ollama.com, puis :
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```
Il faut environ 8 Go de RAM. Si votre ordinateur est plus modeste, essayez `qwen2.5:3b`.

**Option 2 – Google Gemini (offre gratuite).** Créez une clé sur https://aistudio.google.com/apikey,
puis décommentez le bloc Gemini dans `.env` et collez votre clé.
Si un nom de modèle a changé, consultez la liste des modèles sur le site de Google AI.

## Utilisation

```bash
python ingest.py                          # 1. indexe les documents de data/docs
python chat.py                            # 2. pose tes questions à l'agent
python -m evals.run_evals                 # 3. mesure la qualité sur le jeu d'évaluation
python -m evals.run_evals --retrieval-only   # recherche seule, sans le LLM de chat
pytest -q                                 # tests automatisés
```

Après avoir ajouté ou modifié des documents, relancez `python ingest.py`.

## Évaluation

`evals/dataset.json` contient des questions avec la source et les mots-clés attendus, dont une question
**hors sujet** : l'agent doit répondre qu'il ne trouve pas l'information au lieu d'inventer.

Deux scores sont calculés :
- **retrieval_score** : la bonne source fait-elle partie des extraits retrouvés ?
- **answer_score** : la réponse contient-elle l'information attendue ?

`--min-score 0.8` fait échouer le script sous 80 % : on peut l'ajouter à la CI pour bloquer une régression
quand on change de modèle, de prompt ou de taille de chunk.

## Modèle de secours (facultatif)

Avec `CHAT_MODEL_FALLBACK=gemini-3.8-flash-lite` dans `.env`, si le modèle principal est surchargé (erreur 503)
ou a épuisé son quota (erreur 429), l'agent recommence automatiquement la question avec le modèle de secours.
Le basculement est enregistré dans les traces, et le modèle utilisé s'affiche sous chaque réponse.

## Recherche sur Internet (facultatif)

Ajoutez `WEB_SEARCH=on` dans `.env` pour donner à l'agent un troisième outil, `search_web`.
Le LLM choisit alors lui-même la bonne source : les documents internes pour les questions sur
l'entreprise, le web pour les questions générales ou récentes. Les pages web sont citées par leur URL.

Le prompt précise que le contenu des pages web est une simple donnée, jamais une instruction :
c'est une protection contre l'injection de prompt. Sans `WEB_SEARCH=on`, l'outil n'est même pas
proposé au LLM.

## Serveur MCP

`mcp_server.py` expose deux outils, `search_documents` et `list_documents`, via le Model Context Protocol.
Exemple de configuration pour Claude Desktop (`claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "agent-documentaire": {
      "command": "C:/chemin/vers/agent-documentaire/.venv/Scripts/python.exe",
      "args": ["C:/chemin/vers/agent-documentaire/mcp_server.py"],
      "cwd": "C:/chemin/vers/agent-documentaire"
    }
  }
}
```

## Pistes d'amélioration

- Recherche hybride (mots-clés + vecteurs) et *reranking* des extraits.
- Orchestration avec **LangGraph** pour des agents à plusieurs étapes.
- Traces envoyées dans **Langfuse** au lieu d'un fichier local.
- Évaluations avec **promptfoo** ou un « LLM juge » pour noter la qualité des réponses.
- Base vectorielle de production (pgvector, Qdrant, Azure AI Search) et API FastAPI.
