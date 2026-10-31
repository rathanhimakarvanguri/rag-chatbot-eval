"""Streamlit chat UI: `streamlit run app/ui.py` (install with `pip install -e .[ui]`)."""

from __future__ import annotations

import streamlit as st

from ragkit import build_pipeline

st.set_page_config(page_title="Handbook RAG Chatbot", page_icon="💬")
st.title("Handbook RAG Chatbot")
st.caption("Answers come only from the documents in `data/corpus`, with citations. It says so when it can't answer.")

with st.sidebar:
    retriever = st.selectbox("Retriever", ["hybrid", "bm25", "dense"])
    generator = st.selectbox("Generator", ["extractive", "llm"], help="'llm' uses the LLM_* environment variables")
    k = st.slider("Passages (k)", 1, 8, 3)
    min_support = st.slider("Abstention threshold", 0.0, 0.8, 0.30, 0.05)


@st.cache_resource
def load(retriever: str, generator: str, k: int, min_support: float):
    return build_pipeline(retriever=retriever, generator=generator, k=k, min_support=min_support)


pipeline = load(retriever, generator, k, min_support)

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Ask about leave, expenses, IT security, benefits..."):
    st.chat_message("user").markdown(question)
    ans = pipeline.answer(question)
    body = ans.text
    if ans.citations:
        body += "\n\n" + "\n".join(f"[{c.n}] *{c.title} > {c.section}*" for c in ans.citations)
    with st.chat_message("assistant"):
        st.markdown(body)
        with st.expander(f"Retrieved passages (support = {ans.support:.2f})"):
            for h in ans.contexts:
                st.markdown(f"**{h.rank}. {h.chunk.title} > {h.chunk.section}** — score {h.score:.3f}")
                st.write(h.chunk.text)
    st.session_state.history += [("user", question), ("assistant", body)]
