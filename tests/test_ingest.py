from dataclasses import replace

import ingest
from app.config import Settings
from app.store import VectorStore
from tests.fakes import FakeEmbedder


def test_build_index_creates_the_index_used_by_the_web_page(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "conges.md").write_text("Chaque salarié a droit à 25 jours de congés payés.", encoding="utf-8")
    settings = replace(Settings(), docs_dir=str(tmp_path / "docs"), index_path=str(tmp_path / "index.json"))
    monkeypatch.setattr(ingest, "make_embedder", lambda s: FakeEmbedder())

    store = ingest.build_index(settings)

    assert store is not None and store.sources() == ["conges.md"]
    assert VectorStore.load(settings.index_path).sources() == ["conges.md"]


def test_build_index_without_documents_returns_none(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    settings = replace(Settings(), docs_dir=str(tmp_path / "docs"), index_path=str(tmp_path / "index.json"))
    monkeypatch.setattr(ingest, "make_embedder", lambda s: FakeEmbedder())
    assert ingest.build_index(settings) is None
