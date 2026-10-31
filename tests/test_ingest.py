from ragkit.ingest import chunk_markdown, slugify

DOC = """# Title

Preamble that should be dropped.

## First Section

Alpha one. Alpha two.

## Second Section

Beta.
"""


def test_splits_on_h2_and_drops_preamble():
    chunks = chunk_markdown("doc", DOC)
    assert [c.id for c in chunks] == ["doc#first-section", "doc#second-section"]
    assert all("Preamble" not in c.text for c in chunks)
    assert chunks[0].title == "Title"
    assert chunks[0].render().startswith("Title > First Section")


def test_long_sections_are_windowed_with_overlap():
    body = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_markdown("doc", f"# T\n\n## Long\n\n{body}\n", max_chars=200, overlap=1)
    assert len(chunks) > 1
    assert all(len(c.text) <= 230 for c in chunks)
    assert {c.section_id for c in chunks} == {"doc#long"}
    # consecutive windows share a sentence
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_slugify():
    assert slugify("Travel & Expense: Policy!") == "travel-expense-policy"


def test_corpus_loads(chunks):
    assert len(chunks) > 30
    assert len({c.id for c in chunks}) == len(chunks)
