from app.agent import Agent
from tests.fakes import FakeChatClient, FakeEmbedder, FakeWebSearcher, WebChatClient
from tests.test_store import build_store


def make_agent(client, web=None):
    return Agent(client=client, model="fake", store=build_store(), embedder=FakeEmbedder(), web_searcher=web)


def test_web_tool_is_hidden_when_disabled():
    agent = make_agent(FakeChatClient())
    assert "search_web" not in [t["function"]["name"] for t in agent.tools]
    assert "search_web" not in agent.system_prompt


def test_web_tool_is_offered_when_enabled():
    client = WebChatClient()
    make_agent(client, web=FakeWebSearcher()).ask("Qu'est-ce que l'ETNA ?")
    assert "search_web" in client.tools_seen


def test_agent_uses_web_results_and_cites_urls():
    web = FakeWebSearcher()
    answer = make_agent(WebChatClient(), web=web).ask("Qu'est-ce que l'ETNA ?")
    assert web.queries == ["ETNA école"]
    assert answer.sources == ["https://etna.io"]
    assert answer.tool_calls == [{"name": "search_web", "args": {"query": "ETNA école"}}]


def test_web_failure_is_reported_to_the_llm_without_crashing():
    client = WebChatClient()
    answer = make_agent(client, web=FakeWebSearcher(fail=True)).ask("Qu'est-ce que l'ETNA ?")
    tool_message = next(m for m in client.calls[1] if m["role"] == "tool")
    assert "a échoué" in tool_message["content"]
    assert answer.sources == []
