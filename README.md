# RAG Chatbot with an Evaluation Suite

[![CI](https://github.com/rathanhimakarvanguri/rag-chatbot-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/rathanhimakarvanguri/rag-chatbot-eval/actions/workflows/ci.yml)

A retrieval-augmented question-answering bot over an employee handbook that **cites every answer** and
**says "I don't know" instead of guessing** — plus the part most RAG demos skip: a test set and metrics that
show how well it actually works, wired into CI so a regression fails the build.

Built as a portfolio project by a QA engineer moving into AI/ML: the goal was to treat an LLM app like any
other system under test.

```mermaid
flowchart LR
    Q[Question] --> R{Hybrid retriever}
    R -->|BM25| S[Sparse ranks]
    R -->|Embeddings| D[Dense ranks]
    S & D --> F[Reciprocal rank fusion<br/>top-k passages]
    F --> G{Support score<br/>≥ threshold?}
    G -- no --> A0[Abstain:<br/>"I don't know"]
    G -- yes --> L[Generator<br/>LLM or extractive]
    L --> C[Answer + citations]
```

## What's inside

| Piece | Details |
|---|---|
| Ingestion | Markdown split on `##` sections; long sections packed into overlapping sentence windows |
| Retrieval | BM25, dense embeddings (`bge-small` or an offline TF-IDF+SVD fallback), and hybrid via reciprocal rank fusion |
| Abstention gate | IDF-weighted share of the question's words found in the retrieved passages; below the threshold the bot refuses |
| Generation | Any OpenAI-compatible endpoint (OpenAI, Groq, local Ollama) with a cite-or-abstain prompt, or an offline extractive generator |
| Evaluation | 44 labelled questions (27 factual, 9 paraphrased, 8 unanswerable) scored on retrieval, answer quality, grounding and abstention |
| Serving | FastAPI `/ask` endpoint, Streamlit chat UI, Dockerfile |
| Quality | 34 pytest tests incl. eval regression thresholds, ruff, GitHub Actions on Python 3.10 and 3.12 |

The corpus is a **fictional** company handbook (`data/corpus/`), written for this project so every answer can be
checked against a known source. Point `--corpus` at your own Markdown folder to use real documents.

## Results

Offline configuration (no API key, no model download): TF-IDF+SVD embeddings, extractive generator, k=3,
support threshold 0.30. Reproduce with `make eval`; full report in [`reports/results.md`](reports/results.md).

| Retriever | Hit@1 | Hit@3 | MRR | Key-fact recall | Citation precision | Answered unanswerable ↓ | False abstain ↓ |
|---|---|---|---|---|---|---|---|
| BM25 | 0.889 | 0.944 | 0.917 | 0.806 | 0.938 | 0.125 | 0.111 |
| Dense (TF-IDF+SVD) | 0.889 | 0.972 | 0.926 | 0.806 | 0.968 | 0.125 | 0.139 |
| Hybrid (RRF) | 0.889 | 0.944 | 0.917 | 0.806 | 0.938 | 0.125 | 0.111 |

**Abstention gate.** Without it the bot answered all 8 unanswerable questions (hallucinating from the nearest
passage). With it, it answers 1 of 8, at the cost of wrongly refusing 4 of 36 answerable ones. The threshold
sweep in the report shows the trade-off curve.

**Where it fails.** Factual questions are retrieved perfectly (Hit@3 = 1.00), but paraphrased ones drop to
0.78 — e.g. *"I left my work laptop in a taxi"* doesn't share words with *"lost or stolen devices"*. Lexical
embeddings can't bridge that gap, which is exactly what a real embedding model is for; run
`--embedder BAAI/bge-small-en-v1.5` to measure the improvement.

**Honest caveats.** 44 questions is a small set, and the threshold was chosen on the same set it's reported
on. Faithfulness is 1.0 here only because the extractive generator copies sentences verbatim — the metric
becomes meaningful with an LLM generator (`make eval-llm`), where the optional LLM judge adds a second opinion.

### Metrics

- **Hit@k / MRR / nDCG@3** — is the gold section among the top-k passages, and how high
- **Token F1** — word overlap with the reference answer
- **Key-fact recall** — share of the reference's numbers (days, rupee amounts, limits) present in the answer.
  In policy QA a fluent answer with the wrong number is the worst kind of hallucination
- **Faithfulness** — share of answer sentences whose words *and numbers* all appear in the retrieved passages
- **Citation precision** — share of cited passages that are actually the right section
- **Answered unanswerable / false-abstain rate** — the two ways an abstention gate can fail

## Quick start

```bash
git clone https://github.com/rathanhimakarvanguri/rag-chatbot-eval && cd rag-chatbot-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"

make test     # 34 tests
make eval     # writes reports/results.md
make api      # http://localhost:8000/docs
make ui       # Streamlit chat
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Can I fly premium economy on a 10-hour flight?"}'
```

### With an LLM and real embeddings

```bash
cp .env.example .env            # pick OpenAI, Groq or local Ollama
set -a && source .env && set +a
pip install -e ".[models]"      # sentence-transformers

python -m ragkit.eval.run --generator llm --judge --embedder BAAI/bge-small-en-v1.5 --out reports/llm
RAG_GENERATOR=llm RAG_EMBEDDER=BAAI/bge-small-en-v1.5 make api
```

## Project layout

```
src/ragkit/
  ingest.py        # Markdown -> section chunks
  retrievers.py    # BM25, dense, hybrid (RRF)
  embeddings.py    # sentence-transformers or offline TF-IDF+SVD
  pipeline.py      # retrieve -> support gate -> generate -> citations
  llm.py           # OpenAI-compatible client, extractive fallback, prompt
  eval/metrics.py  # retrieval, answer, grounding, abstention metrics, LLM judge
  eval/run.py      # CLI: compare configs, threshold sweep, Markdown/JSON report
app/api.py         # FastAPI
app/ui.py          # Streamlit
data/eval/qa.jsonl # labelled questions with gold sections
```

## Ideas for next steps

- Cross-encoder reranking (e.g. `bge-reranker-base`) on the fused top-20
- Query rewriting for paraphrased questions
- A larger, held-out eval split so the abstention threshold is tuned and reported on different questions

## License

MIT
