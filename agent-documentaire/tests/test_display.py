from app.display import format_answer

ANSWER = """**Auchan** est une entreprise française [https://fr.wikipedia.org/wiki/Auchan].

Quelques repères clés :
* **Siège social :** Croix, près de Roubaix [https://fr.wikipedia.org/wiki/Auchan].
* **Activité :** grande distribution [https://fr.wikipedia.org/wiki/Auchan, https://grokipedia.com/page/Auchan]."""

RETRIEVED = ["ux4mation.md#1", "notes_de_frais.md#0", "https://fr.wikipedia.org/wiki/Auchan", "https://grokipedia.com/page/Auchan"]


def render(text=ANSWER, sources=RETRIEVED):
    return format_answer(text, sources, "gemini-test", 27444, 3130, color=False, width=80)


def test_citations_become_numbers():
    out = render()
    assert "[https://" not in out.split("Sources")[0]
    assert "entreprise française [1]." in out
    assert "grande distribution [1][2]." in out


def test_only_cited_sources_are_listed():
    out = render()
    sources_part = out.split("Sources")[1]
    assert "[1] https://fr.wikipedia.org/wiki/Auchan" in sources_part
    assert "[2] https://grokipedia.com/page/Auchan" in sources_part
    assert "ux4mation.md" not in sources_part


def test_markdown_is_cleaned():
    out = render()
    assert "**" not in out
    assert "   • Siège social : Croix" in out


def test_document_citations_are_numbered_too():
    out = render("Deux jours par semaine [teletravail.md#0].", ["teletravail.md#0", "conges.md#1"])
    assert "Deux jours par semaine [1]." in out
    assert "[1] teletravail.md#0" in out
    assert "conges.md#1" not in out


def test_without_citations_retrieved_sources_are_shown():
    out = render("Je ne trouve pas cette information.", ["conges.md#1"])
    assert "Extraits consultés" in out
    assert "[1] conges.md#1" in out


def test_normal_brackets_are_left_alone():
    out = render("Le tableau [voir ci-dessous] est utile.", [])
    assert "[voir ci-dessous]" in out


def test_metadata_line():
    assert "gemini-test · 27.4 s · 3130 tokens" in render()
