from app.loader import Chunk, load_documents
from app.store import VectorStore
from tests.fakes import FakeEmbedder


def build_store():
    chunks = load_documents("data/docs", size=800, overlap=150)
    store = VectorStore()
    store.add(chunks, FakeEmbedder().embed([c.text for c in chunks]))
    return store


def test_search_finds_the_right_document():
    store = build_store()
    query = FakeEmbedder().embed(["indemnité de télétravail par jour"])[0]
    best_chunk, score = store.search(query, k=1)[0]
    assert best_chunk.source == "teletravail.md"
    assert 0 < score <= 1


def test_search_returns_at_most_k_results():
    store = build_store()
    query = FakeEmbedder().embed(["congés"])[0]
    assert len(store.search(query, k=2)) == 2


def test_empty_store_returns_nothing():
    assert VectorStore().search([1.0, 0.0], k=3) == []


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add([Chunk("a#0", "a", "texte A"), Chunk("b#0", "b", "texte B")], [[1, 0], [0, 1]])
    path = tmp_path / "index.json"
    store.save(str(path))

    loaded = VectorStore.load(str(path))
    assert len(loaded) == 2
    assert loaded.search([0, 1], k=1)[0][0].id == "b#0"
