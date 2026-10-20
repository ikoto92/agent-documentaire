"""Lecture des documents et découpage en morceaux (chunks)."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


@dataclass
class Chunk:
    id: str  # ex. "conges.md#2"
    source: str  # nom du fichier d'origine
    text: str


def read_file(path: Path) -> str:
    """Renvoie le texte brut d'un fichier PDF, Markdown ou texte."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Découpe un texte en morceaux d'environ `size` caractères.

    Les morceaux se chevauchent de `overlap` caractères pour ne pas couper
    une information importante en deux. On coupe de préférence sur un espace.
    """
    if size <= 0:
        raise ValueError("size doit être positif")
    if not 0 <= overlap < size:
        raise ValueError("overlap doit être compris entre 0 et size")

    text = " ".join(text.split())  # normalise les espaces et retours à la ligne
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = end - overlap
        # on recommence au début d'un mot, pas au milieu
        if 0 < next_start and text[next_start - 1] != " ":
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = max(next_start, start + 1)  # garantit que l'on avance
    return [c for c in chunks if c]


def load_documents(docs_dir: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Lit tous les documents d'un dossier et renvoie la liste de leurs chunks."""
    folder = Path(docs_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {docs_dir}")

    chunks: list[Chunk] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        for i, piece in enumerate(split_text(read_file(path), size, overlap)):
            chunks.append(Chunk(id=f"{path.name}#{i}", source=path.name, text=piece))
    return chunks
