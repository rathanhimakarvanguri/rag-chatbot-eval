import pytest

from ragkit.retrievers import build_retriever


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid"])
@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many sick leave days per year?", "leave-policy#sick-leave"),
        ("What is the minimum password length?", "it-security#passwords-and-authentication"),
        ("hotel limit per night in metro cities", "expenses#hotels-and-meals"),
    ],
)
def test_finds_obvious_section(chunks, name, question, expected):
    hits = build_retriever(name, chunks).search(question, k=3)
    assert expected in [h.chunk.section_id for h in hits]


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid"])
def test_ranks_are_ordered_and_unique(chunks, name):
    hits = build_retriever(name, chunks).search("annual leave carry forward", k=5)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert len({h.chunk.id for h in hits}) == 5
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_unknown_retriever(chunks):
    with pytest.raises(ValueError):
        build_retriever("magic", chunks)
