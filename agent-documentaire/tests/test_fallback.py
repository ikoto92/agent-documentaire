import json

import pytest

from app.agent import Agent
from app.tracing import Tracer
from tests.fakes import FakeEmbedder, OverloadedClient, ProviderError
from tests.test_store import build_store


def make_agent(client, fallbacks, tracer=None):
    return Agent(client=client, model="principal", store=build_store(), embedder=FakeEmbedder(),
                 fallback_models=fallbacks, tracer=tracer)


@pytest.mark.parametrize("status", [503, 429])
def test_switches_to_backup_model(status):
    client = OverloadedClient(status)
    answer = make_agent(client, ["secours"]).ask("Combien de jours de congés payés ?")
    assert answer.model == "secours"
    assert answer.text.startswith("Vous avez 25 jours")
    assert client.models_called[0] == "principal"


def test_error_is_raised_without_backup_model():
    with pytest.raises(ProviderError):
        make_agent(OverloadedClient(503), []).ask("Question")


def test_other_errors_are_not_hidden():
    with pytest.raises(ProviderError):
        make_agent(OverloadedClient(400), ["secours"]).ask("Question")


def test_fallback_is_traced(tmp_path):
    trace = tmp_path / "t.jsonl"
    make_agent(OverloadedClient(503), ["secours"], Tracer(str(trace))).ask("Combien de jours de congés payés ?")
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert events[0] == {"timestamp": events[0]["timestamp"], "event": "fallback", "from": "principal", "to": "secours", "status": 503}
    assert events[1]["model"] == "secours"


class APITimeoutError(Exception):
    """Même nom que l'erreur de délai dépassé de la bibliothèque openai."""


class SlowClient(OverloadedClient):
    def _create(self, model, messages, tools=None, temperature=0):
        self.models_called.append(model)
        if model == "principal":
            raise APITimeoutError("trop lent")
        return super(OverloadedClient, self)._create(model, messages, tools, temperature)


def test_timeout_also_switches_to_backup_model():
    client = SlowClient()
    answer = make_agent(client, ["secours"]).ask("Combien de jours de congés payés ?")
    assert answer.model == "secours"
