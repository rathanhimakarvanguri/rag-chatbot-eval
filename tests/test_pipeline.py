import httpx
import pytest

from ragkit import build_pipeline
from ragkit.llm import ABSTAIN, OpenAICompatibleLLM, is_abstention, parse_citations


@pytest.fixture(scope="module")
def pipeline():
    return build_pipeline(min_support=0.30)


def test_answers_with_citation_and_correct_fact(pipeline):
    ans = pipeline.answer("How many days of sick leave do employees get?")
    assert not ans.abstained
    assert "12" in ans.text
    assert ans.citations and ans.citations[0].chunk_id == "leave-policy#sick-leave"


def test_abstains_on_out_of_scope_question(pipeline):
    ans = pipeline.answer("What is the company's stock option vesting schedule?")
    assert ans.abstained
    assert ans.text == ABSTAIN
    assert ans.citations == []


def test_parse_citations_dedupes_in_order():
    assert parse_citations("A [2]. B [1][2]. C [3]") == [2, 1, 3]


def test_is_abstention():
    assert is_abstention(ABSTAIN)
    assert not is_abstention("24 days [1]")


def test_llm_client_sends_context_and_parses_reply(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, body=json)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "12 days [1]"}}]}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OpenAICompatibleLLM(model="test-model", base_url="http://localhost:11434/v1", api_key="")
    out = llm.generate("sick leave?", ["Leave > Sick Leave\n12 days of paid sick leave."])
    assert out == "12 days [1]"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert "[1] Leave > Sick Leave" in captured["body"]["messages"][1]["content"]
    assert captured["body"]["temperature"] == 0.0


def test_pipeline_uses_llm_citations(monkeypatch, pipeline):
    class FakeLLM:
        name = "fake"

        def generate(self, question, passages):
            return "Laptops are refreshed every 3 years for engineering roles. [1]"

    monkeypatch.setattr(pipeline, "generator", FakeLLM())
    ans = pipeline.answer("How often are engineering laptops refreshed?")
    assert ans.citations[0].n == 1
    assert ans.citations[0].chunk_id == ans.contexts[0].chunk.id
