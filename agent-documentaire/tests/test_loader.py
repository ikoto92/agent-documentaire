import pytest

from app.loader import load_documents, split_text


def test_split_short_text_gives_one_chunk():
    assert split_text("Bonjour tout le monde", size=100, overlap=10) == ["Bonjour tout le monde"]


def test_split_respects_size_and_overlaps():
    text = " ".join(f"mot{i}" for i in range(300))
    chunks = split_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # le début d'un chunk reprend la fin du précédent (chevauchement)
    assert chunks[1].split()[0] in chunks[0]


def test_split_covers_the_whole_text():
    words = [f"mot{i}" for i in range(500)]
    chunks = split_text(" ".join(words), size=120, overlap=30)
    assert set(" ".join(chunks).split()) == set(words)


def test_split_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("abc", size=10, overlap=10)


def test_load_documents_reads_md_and_ignores_other_files(tmp_path):
    (tmp_path / "a.md").write_text("Premier document.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Deuxième document.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    chunks = load_documents(str(tmp_path))
    assert [c.id for c in chunks] == ["a.md#0", "b.txt#0"]


def test_load_documents_missing_folder():
    with pytest.raises(FileNotFoundError):
        load_documents("dossier/inexistant")
