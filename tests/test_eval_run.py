from ragkit.eval.run import main


def test_eval_cli_writes_report(tmp_path):
    results = main(["--retrievers", "bm25", "--out", str(tmp_path)])
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "predictions-bm25.jsonl").exists()
    gated = results["configs"][0]["gated"]
    # Regression guards: fail CI if a change makes retrieval or abstention clearly worse.
    assert gated["hit@3"] >= 0.90
    assert gated["answered_unanswerable"] <= 0.25
    assert gated["key_fact_recall"] >= 0.70
