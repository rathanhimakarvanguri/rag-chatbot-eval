"""FastAPI service: `uvicorn app.api:app --reload`, then open http://localhost:8000/docs."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ragkit import build_pipeline

app = FastAPI(title="RAG Chatbot", version="0.1.0")


@lru_cache(maxsize=1)
def get_pipeline():
    return build_pipeline(
        corpus=os.getenv("RAG_CORPUS", "data/corpus"),
        retriever=os.getenv("RAG_RETRIEVER", "hybrid"),
        generator=os.getenv("RAG_GENERATOR", "extractive"),
        embedder=os.getenv("RAG_EMBEDDER", "lsa"),
        k=int(os.getenv("RAG_K", "3")),
        min_support=float(os.getenv("RAG_MIN_SUPPORT", "0.30")),
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["How many sick leave days do I get?"])


class CitationOut(BaseModel):
    n: int
    chunk_id: str
    title: str
    section: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    support: float
    citations: list[CitationOut]


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {"status": "ok", "retriever": p.retriever.name, "generator": p.generator.name}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    ans = get_pipeline().answer(req.question)
    return AskResponse(
        answer=ans.text,
        abstained=ans.abstained,
        support=round(ans.support, 3),
        citations=[CitationOut(**c.__dict__) for c in ans.citations],
    )
