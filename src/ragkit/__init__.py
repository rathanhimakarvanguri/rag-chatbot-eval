"""ragkit: a small, testable retrieval-augmented QA pipeline with an evaluation harness."""

from ragkit.ingest import Chunk, load_corpus
from ragkit.pipeline import Answer, RAGPipeline, build_pipeline

__all__ = ["Answer", "Chunk", "RAGPipeline", "build_pipeline", "load_corpus"]
__version__ = "0.1.0"
