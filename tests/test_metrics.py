import pytest

from ragkit.eval import metrics as M


def test_retrieval_metrics():
    retrieved, gold = ["a", "b", "c"], ["b"]
    assert M.hit_at_k(retrieved, gold, 1) == 0
    assert M.hit_at_k(retrieved, gold, 2) == 1
    assert M.reciprocal_rank(retrieved, gold) == 0.5
    assert M.recall_at_k(retrieved, ["b", "z"], 3) == 0.5
    assert M.ndcg_at_k(["b", "a"], gold, 2) == pytest.approx(1.0)
    assert 0 < M.ndcg_at_k(["a", "b"], gold, 2) < 1


def test_token_f1_ignores_citation_markers():
    assert M.token_f1("24 days per year [1]", "24 days per year") == pytest.approx(1.0)
    assert M.token_f1("nothing alike", "24 days") == 0.0


def test_key_fact_recall_catches_wrong_numbers():
    ref = "Up to INR 50,000; above that the finance controller must approve."
    assert M.key_fact_recall("Managers approve up to INR 50,000 [1]", ref) == 1.0
    assert M.key_fact_recall("Managers approve up to INR 25,000 [1]", ref) == 0.0
    assert M.key_fact_recall("anything", "no numbers here") == 1.0


def test_faithfulness_flags_unsupported_numbers_and_claims():
    ctx = ["Employees receive 12 days of paid sick leave per year."]
    assert M.faithfulness("Employees receive 12 days of paid sick leave per year. [1]", ctx) == 1.0
    assert M.faithfulness("Employees receive 20 days of paid sick leave per year.", ctx) == 0.0
    mixed = "Employees receive 12 days of sick leave. Unused days are paid out in cash every March."
    assert M.faithfulness(mixed, ctx) == 0.5


def test_citation_precision():
    assert M.citation_precision(["a", "b"], ["a"]) == 0.5
    assert M.citation_precision([], ["a"]) == 0.0


def test_abstention_report():
    answerable = [True, True, False, False]
    abstained = [False, True, True, False]
    r = M.abstention_report(answerable, abstained)
    assert r["abstain_precision"] == 0.5
    assert r["abstain_recall"] == 0.5
    assert r["false_abstain_rate"] == 0.5


def test_llm_judge_parses_number():
    class Fake:
        def chat(self, messages):
            return "Score: 0.75"

    assert M.llm_judge_faithfulness(Fake(), "a", ["b"]) == 0.75
