"""Mise en forme des réponses de l'agent pour le terminal.

- Les citations [url] ou [document#n] deviennent des numéros courts : [1], [2]...
- Les sources réellement citées sont listées à la fin, numérotées.
- Le Markdown simple est rendu : **gras**, listes à puces, titres.
- Le texte est coupé proprement à la largeur de la fenêtre.
"""

import re
import shutil
import textwrap

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_SOURCE = re.compile(r"^(https?://\S+|[\w.-]+#\d+)$")


def _number_citations(text: str) -> tuple[str, list[str]]:
    """Remplace chaque citation par des numéros et renvoie la liste ordonnée des sources citées."""
    cited: list[str] = []

    def replace(match: re.Match) -> str:
        parts = [p.strip() for p in match.group(1).split(",")]
        if not parts or not all(_SOURCE.match(p) for p in parts):
            return match.group(0)  # ce n'est pas une citation : on laisse tel quel
        numbers = []
        for p in parts:
            if p not in cited:
                cited.append(p)
            numbers.append(f"[{cited.index(p) + 1}]")
        return "".join(numbers)

    return _CITATION.sub(replace, text), cited


def _style(line: str, color: bool) -> str:
    if not color:
        return line.replace("**", "")
    return re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", line)


def format_answer(text: str, retrieved_sources: list[str], model: str, latency_ms: float,
                  tokens: int, color: bool = True, width: int | None = None) -> str:
    width = width or min(shutil.get_terminal_size((100, 20)).columns, 110)
    body, cited = _number_citations(text.strip())

    lines: list[str] = []
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        bullet = re.match(r"^[*\-•]\s+(.*)", stripped)
        title = re.match(r"^#{1,6}\s+(.*)", stripped)
        if bullet:
            wrapped = textwrap.wrap(bullet.group(1), width - 6) or [""]
            lines.append("   • " + wrapped[0])
            lines.extend("     " + w for w in wrapped[1:])
        elif title:
            lines.append(f"**{title.group(1)}**")
        else:
            lines.extend("  " + w for w in textwrap.wrap(stripped, width - 2))

    out = [_style(line, color) for line in lines]

    sources = cited or retrieved_sources
    if sources:
        label = "Sources" if cited else "Extraits consultés"
        out += ["", (f"{BOLD}{label}{RESET}" if color else label)]
        for i, s in enumerate(sources, 1):
            out.append(f"  [{i}] " + (f"{CYAN}{s}{RESET}" if color else s))

    meta = f"{model} · {latency_ms / 1000:.1f} s · {tokens} tokens"
    out += ["", (f"{DIM}{meta}{RESET}" if color else meta)]
    return "\n".join(out)
