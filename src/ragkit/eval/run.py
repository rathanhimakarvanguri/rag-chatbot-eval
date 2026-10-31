"""Evaluate retrievers x generator on a labelled QA set and write a Markdown + JSON report.

python -m ragkit.eval.run                                  # offline: bm25/dense/hybrid + extractive
python -m ragkit.eval.run --generator llm --judge          # with an LLM (set LLM_* env vars)
python -m ragkit.eval.run --embedder BAAI/bge-small-en-v1.5  # real embeddings
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from ragkit.embeddings import get_embedder
from ragkit.eval import metrics as M
from ragkit.ingest import load_corpus
from ragkit.llm import ABSTAIN, OpenAICompatibleLLM, get_generator
from ragkit.pipeline import RAGPipeline, SupportScorer
from ragkit.retrievers import build_retriever


@dataclass
class Example:
    id: str
    type: str
    question: str
    answer: str
    gold: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def load_examples(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]


def run_config(pipeline: RAGPipeline, examples: list[Example], judge=None) -> list[dict]:
    """Run every example with the gate disabled; the gate is applied afterwards so one run
    can be scored at many thresholds."""
    records = []
    for ex in examples:
        t0 = time.perf_counter()
        ans = pipeline.answer(ex.question)
        latency = time.perf_counter() - t0
        retrieved = [h.chunk.section_id for h in ans.contexts]
        contexts = [h.chunk.render() for h in ans.contexts]
        rec = {
            "id": ex.id,
            "type": ex.type,
            "answerable": ex.answerable,
            "question": ex.question,
            "reference": ex.answer,
            "prediction": ans.text,
            "generator_abstained": ans.abstained,
            "support": round(ans.support, 4),
            "retrieved": retrieved,
            "cited": [c.chunk_id.split("--")[0] for c in ans.citations],
            "latency_ms": round(latency * 1000, 1),
            "faithfulness": M.faithfulness(ans.text, contexts) if not ans.abstained else None,
        }
        if judge is not None and not ans.abstained:
            rec["judge_faithfulness"] = M.llm_judge_faithfulness(judge, ans.text, contexts)
        if ex.answerable:
            rec.update(
                hit1=M.hit_at_k(retrieved, ex.gold, 1),
                hit3=M.hit_at_k(retrieved, ex.gold, 3),
                mrr=M.reciprocal_rank(retrieved, ex.gold),
                ndcg3=M.ndcg_at_k(retrieved, ex.gold, 3),
            )
        records.append(rec)
    return records


def gated(rec: dict, threshold: float) -> tuple[bool, str]:
    abstained = rec["generator_abstained"] or rec["support"] < threshold
    return abstained, (ABSTAIN if abstained else rec["prediction"])


def score(records: list[dict], examples: list[Example], threshold: float) -> dict[str, float]:
    gold = {ex.id: ex for ex in examples}
    answerable = [r for r in records if r["answerable"]]
    flags = [gated(r, threshold)[0] for r in records]
    out = {
        "hit@1": statistics.mean(r["hit1"] for r in answerable),
        "hit@3": statistics.mean(r["hit3"] for r in answerable),
        "mrr": statistics.mean(r["mrr"] for r in answerable),
        "ndcg@3": statistics.mean(r["ndcg3"] for r in answerable),
    }
    answered = [r for r, a in zip(records, flags) if r["answerable"] and not a]
    # Answer quality is scored over ALL answerable questions: a wrong abstention scores 0.
    out["token_f1"] = statistics.mean(
        0.0 if gated(r, threshold)[0] else M.token_f1(r["prediction"], r["reference"]) for r in answerable
    )
    out["key_fact_recall"] = statistics.mean(
        0.0 if gated(r, threshold)[0] else M.key_fact_recall(r["prediction"], r["reference"]) for r in answerable
    )
    out["citation_precision"] = (
        statistics.mean(M.citation_precision(r["cited"], gold[r["id"]].gold) for r in answered) if answered else 0.0
    )
    answered_all = [r for r, a in zip(records, flags) if not a]
    out["faithfulness"] = statistics.mean(r["faithfulness"] for r in answered_all) if answered_all else 1.0
    if any("judge_faithfulness" in r for r in answered_all):
        out["judge_faithfulness"] = statistics.mean(
            r["judge_faithfulness"] for r in answered_all if "judge_faithfulness" in r
        )
    # Hallucination proxy: answered an unanswerable question.
    unanswerable = [(r, a) for r, a in zip(records, flags) if not r["answerable"]]
    out["answered_unanswerable"] = sum(not a for _, a in unanswerable) / max(1, len(unanswerable))
    out.update(M.abstention_report([r["answerable"] for r in records], flags))
    out["p50_latency_ms"] = statistics.median(r["latency_ms"] for r in records)
    return out


def _fmt(v: float) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Evaluation report",
        "",
        f"- Corpus: `{results['corpus']}` ({results['n_chunks']} chunks)",
        f"- Questions: {results['n_questions']} ({results['n_answerable']} answerable, "
        f"{results['n_questions'] - results['n_answerable']} unanswerable)",
        f"- Generator: `{results['generator']}`, embedder: `{results['embedder']}`, k={results['k']}, "
        f"support threshold={results['threshold']}",
        "",
        "## Retrievers compared (gate on)",
        "",
    ]
    keys = [
        "hit@1",
        "hit@3",
        "mrr",
        "ndcg@3",
        "token_f1",
        "key_fact_recall",
        "citation_precision",
        "faithfulness",
        "answered_unanswerable",
        "abstain_f1",
        "false_abstain_rate",
    ]
    if any("judge_faithfulness" in c["gated"] for c in results["configs"]):
        keys.insert(8, "judge_faithfulness")
    lines.append("| retriever | " + " | ".join(keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for cfg in results["configs"]:
        lines.append(f"| {cfg['retriever']} | " + " | ".join(_fmt(cfg["gated"][k]) for k in keys) + " |")
    lines += [
        "",
        "## Effect of the abstention gate",
        "",
        "| retriever | gate | token_f1 | answered_unanswerable | false_abstain_rate |",
        "|---|---|---|---|---|",
    ]
    for cfg in results["configs"]:
        for label, s in (("off", cfg["ungated"]), ("on", cfg["gated"])):
            lines.append(
                f"| {cfg['retriever']} | {label} | {_fmt(s['token_f1'])} | "
                f"{_fmt(s['answered_unanswerable'])} | {_fmt(s['false_abstain_rate'])} |"
            )
    best = results["configs"][-1]
    lines += [
        "",
        f"## Threshold sweep ({best['retriever']})",
        "",
        "| threshold | abstain_f1 | answered_unanswerable | false_abstain_rate | token_f1 |",
        "|---|---|---|---|---|",
    ]
    for row in best["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {_fmt(row['abstain_f1'])} | {_fmt(row['answered_unanswerable'])} | "
            f"{_fmt(row['false_abstain_rate'])} | {_fmt(row['token_f1'])} |"
        )
    lines += [
        "",
        f"## By question type ({best['retriever']})",
        "",
        "| type | n | hit@1 | hit@3 | token_f1 |",
        "|---|---|---|---|---|",
    ]
    for t, s in best["by_type"].items():
        lines.append(f"| {t} | {s['n']} | {_fmt(s['hit@1'])} | {_fmt(s['hit@3'])} | {_fmt(s['token_f1'])} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--qa", default="data/eval/qa.jsonl")
    p.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    p.add_argument("--generator", default="extractive", choices=["extractive", "llm"])
    p.add_argument("--embedder", default="lsa")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--judge", action="store_true", help="also score faithfulness with an LLM judge")
    p.add_argument("--out", default="reports")
    args = p.parse_args(argv)

    chunks = load_corpus(args.corpus)
    examples = load_examples(args.qa)
    judge = OpenAICompatibleLLM() if args.judge else None
    results = {
        "corpus": args.corpus,
        "n_chunks": len(chunks),
        "n_questions": len(examples),
        "n_answerable": sum(e.answerable for e in examples),
        "generator": args.generator,
        "embedder": args.embedder,
        "k": args.k,
        "threshold": args.threshold,
        "configs": [],
    }
    for name in args.retrievers:
        pipeline = RAGPipeline(
            build_retriever(name, chunks, get_embedder(args.embedder)),
            get_generator(args.generator),
            SupportScorer(chunks),
            k=args.k,
            min_support=0.0,
        )
        records = run_config(pipeline, examples, judge)
        by_type = {}
        for t in sorted({e.type for e in examples if e.answerable}):
            sub = [r for r in records if r["type"] == t]
            s = score(sub, examples, args.threshold)
            by_type[t] = {"n": len(sub), "hit@1": s["hit@1"], "hit@3": s["hit@3"], "token_f1": s["token_f1"]}
        cfg = {
            "retriever": pipeline.retriever.name,
            "gated": score(records, examples, args.threshold),
            "ungated": score(records, examples, 0.0),
            "sweep": [{"threshold": t / 20, **score(records, examples, t / 20)} for t in range(0, 13)],
            "by_type": by_type,
        }
        results["configs"].append(cfg)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        slug = name if args.generator == "extractive" else f"{name}-{args.generator}"
        with open(out / f"predictions-{slug}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        g = cfg["gated"]
        print(
            f"{cfg['retriever']:<24} hit@3={g['hit@3']:.3f} mrr={g['mrr']:.3f} f1={g['token_f1']:.3f} "
            f"facts={g['key_fact_recall']:.3f} faithful={g['faithfulness']:.3f} "
            f"abstain_f1={g['abstain_f1']:.3f}"
        )
    write_report(results, Path(args.out))
    print(f"Report written to {Path(args.out) / 'results.md'}")
    return results


if __name__ == "__main__":
    main()
