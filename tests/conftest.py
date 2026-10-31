import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # the API and defaults use paths relative to the repo root


@pytest.fixture(scope="session")
def chunks():
    from ragkit.ingest import load_corpus

    return load_corpus(ROOT / "data" / "corpus")
