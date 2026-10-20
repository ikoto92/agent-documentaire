"""Test de l'interface web avec un faux agent (aucun appel à un vrai modèle)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import app.factory  # noqa: E402
from app.agent import AgentAnswer  # noqa: E402


class FakeAgent:
    def ask(self, question: str) -> AgentAnswer:
        return AgentAnswer(
            text="Vous pouvez télétravailler 2 jours par semaine [teletravail.md#0].",
            sources=["teletravail.md#0", "conges.md#1"],
            latency_ms=1200,
            model="faux-modele",
        )


@pytest.fixture
def web_app(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("INDEX_PATH", str(index))
    monkeypatch.setenv("MAX_QUESTIONS_PER_SESSION", "1")
    monkeypatch.setattr(app.factory, "make_agent", lambda settings=None: FakeAgent())
    return AppTest.from_file("../app_web.py", default_timeout=30)


def test_question_shows_answer_with_numbered_source(web_app):
    web_app.run()
    web_app.chat_input[0].set_value("Combien de jours de télétravail ?").run()

    page = " ".join(m.value for m in web_app.markdown)
    assert "2 jours par semaine [1]." in page
    assert "`teletravail.md#0`" in page
    assert "conges.md#1" not in page  # seules les sources citées sont listées
    assert not web_app.exception


def test_question_limit_protects_the_quota(web_app):
    web_app.run()
    web_app.chat_input[0].set_value("Une question").run()
    assert len(web_app.chat_input) == 0
    assert "limite de 1 question" in web_app.info[0].value
