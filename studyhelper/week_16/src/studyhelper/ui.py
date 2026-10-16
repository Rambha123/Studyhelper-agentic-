import os

import requests
import streamlit as st

API = os.getenv("API_BASE", "http://localhost:8080")
st.set_page_config(page_title="StudyHelper", page_icon="📚")
st.title("📚 StudyHelper")


def call(path: str, payload: dict | None = None):
    try:
        r = requests.post(f"{API}{path}", json=payload or {}, timeout=90)
        if r.status_code == 429:
            st.warning(f"Slow down — retry in {r.headers.get('Retry-After', '?')}s.")
        elif r.status_code >= 400:
            st.error(r.json().get("detail", "Something went wrong."))
        else:
            return r.json()
    except requests.RequestException:
        st.error("Can't reach the backend. Is the API running?")
    return None


chat_tab, notes_tab, card_tab = st.tabs(["Chat", "Notes", "Study card"])

with chat_tab:
    st.session_state.setdefault("history", [])
    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])
    if prompt := st.chat_input("Ask about your notes, do a calculation, or plan your revision"):
        st.chat_message("user").write(prompt)
        res = call("/chat", {"message": prompt, "history": st.session_state.history})
        if res:
            st.chat_message("assistant").write(res["answer"])
            if res["tool_trace"]:
                with st.expander("Tool calls"):
                    st.json(res["tool_trace"])
            st.session_state.history += [{"role": "user", "content": prompt},
                                         {"role": "assistant", "content": res["answer"]}]

with notes_tab:
    st.write("Index the notes in `data/notes/`, or paste new ones.")
    if st.button("Index notes folder"):
        if res := call("/ingest"):
            st.success(f"Indexed {res['chunks_indexed']} chunks")
    name = st.text_input("Note name", "my_notes.txt")
    text = st.text_area("Paste notes", height=200)
    if st.button("Add pasted notes") and text:
        if res := call("/ingest", {"name": name, "text": text}):
            st.success(f"Indexed {res['chunks_indexed']} chunks")

with card_tab:
    topic = st.text_input("Topic")
    if st.button("Generate study card") and topic:
        if card := call("/study-card", {"topic": topic}):
            st.subheader(card["topic"])
            st.write(card["summary"])
            st.markdown("**Key points**\n" + "\n".join(f"- {p}" for p in card["key_points"]))
            st.markdown("**Practice questions**\n" + "\n".join(f"1. {q}" for q in card["practice_questions"]))
            st.caption(f"Difficulty: {card['difficulty']}")
