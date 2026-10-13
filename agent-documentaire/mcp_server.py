"""Bonus : expose la recherche documentaire comme un serveur MCP.

N'importe quel client MCP (Claude Desktop, Claude Code, un IDE...) peut alors
utiliser l'outil search_documents sur vos documents.

Usage : python mcp_server.py   (transport stdio)
"""

from mcp.server.mcpserver import MCPServer

from app.config import get_settings
from app.factory import make_embedder
from app.store import VectorStore

settings = get_settings()
store = VectorStore.load(settings.index_path)
embedder = make_embedder(settings)

mcp = MCPServer("agent-documentaire")


@mcp.tool()
def search_documents(query: str, k: int = 4) -> str:
    """Recherche les passages les plus pertinents dans les documents indexés."""
    results = store.search(embedder.embed([query])[0], k)
    if not results:
        return "Aucun extrait trouvé."
    return "\n\n".join(f"[{c.id}] (score {s:.2f})\n{c.text}" for c, s in results)


@mcp.tool()
def list_documents() -> list[str]:
    """Liste les documents disponibles."""
    return store.sources()


if __name__ == "__main__":
    mcp.run()
