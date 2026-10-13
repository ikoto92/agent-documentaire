import json

from app.agent import Agent
from app.tracing import Tracer
from tests.fakes import FakeChatClient, FakeEmbedder, LoopingChatClient
from tests.test_store import build_store


def make_agent(client, tracer=None):
    return Agent(client=client, model="fake", store=build_store(), embedder=FakeEmbedder(), top_k=3, tracer=tracer)


def test_agent_calls_search_then_answers():
    client = FakeChatClient()
    answer = make_agent(client).ask("Combien de jours de congés payés ?")

    assert answer.text.startswith("Vous avez 25 jours")
    assert answer.tool_calls == [{"name": "search_documents", "args": {"query": "Combien de jours de congés payés ?"}}]
    assert any(s.startswith("conges.md") for s in answer.sources)
    assert answer.usage == {"prompt_tokens": 20, "completion_tokens": 10}


def test_tool_result_is_sent_back_to_the_llm():
    client = FakeChatClient()
    make_agent(client).ask("Combien de jours de congés payés ?")

    second_call = client.calls[1]
    tool_message = next(m for m in second_call if m["role"] == "tool")
    assert "[conges.md#" in tool_message["content"]


def test_agent_stops_after_max_steps():
    answer = make_agent(LoopingChatClient()).ask("Question sans fin")
    assert "nombre d'étapes" in answer.text
    assert len(answer.tool_calls) == 5


def test_each_question_is_traced(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    make_agent(FakeChatClient(), tracer=Tracer(str(trace_file))).ask("Combien de jours de congés payés ?")

    record = json.loads(trace_file.read_text(encoding="utf-8").strip())
    assert record["question"] == "Combien de jours de congés payés ?"
    assert record["prompt_tokens"] == 20
    assert "latency_ms" in record
