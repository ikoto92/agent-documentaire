from evals.run_evals import keywords_ok, retrieval_ok


def test_keywords_all_must_match_by_default():
    assert keywords_ok("Vous avez 25 jours", ["25", "jours"])
    assert not keywords_ok("Vous avez 25 semaines", ["25", "jours"])


def test_keywords_any_mode():
    assert keywords_ok("Je ne trouve pas cette information.", ["ne trouve pas", "aucune information"], "any")


def test_retrieval_ok():
    assert retrieval_ok(["conges.md#0", "teletravail.md#1"], "conges.md")
    assert not retrieval_ok(["teletravail.md#1"], "conges.md")
    assert retrieval_ok([], None)
