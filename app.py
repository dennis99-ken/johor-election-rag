"""
Step 4 - Chat page (Streamlit).

Run:  streamlit run app.py      (opens http://localhost:8501 in your browser)
"""
from datetime import datetime

import streamlit as st

import config
from rag import _load_index, answer
from retrieval import load_chunks

st.set_page_config(page_title="Johor Election 2026 Q&A", page_icon="🗳️")

try:
    _load_index()          # also checks that passages and embeddings belong together
except (FileNotFoundError, RuntimeError) as error:
    st.error(f"The search index isn't ready: {error}  \nRun `python scrape.py` and then `python build_index.py`.")
    st.stop()


@st.cache_data
def coverage():
    chunks = load_chunks()
    dates = sorted(c["published"][:10] for c in chunks)
    nice = lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    return len({c["url"] for c in chunks}), nice(dates[0]), nice(dates[-1])


n_articles, first_date, last_date = coverage()
EXAMPLES = [
    "How many seats did Barisan Nasional win?",
    "What issues did candidates campaign on in Johor Baru?",
    "What was the early voting turnout?",
    "Why did Pakatan Harapan lose in Johor?",
    "Who won the Melaka state election?",          # out of scope: shows the polite decline
]

# ---------- Sidebar: example questions and settings
with st.sidebar:
    st.header("Try a question")
    clicks = [st.button(q, key=f"example_{i}") for i, q in enumerate(EXAMPLES)]
    clicked = next((q for q, c in zip(EXAMPLES, clicks) if c), None)
    st.caption("The last example is outside the coverage, to show how the assistant declines.")
    st.divider()
    show_details = st.checkbox("Show retrieval details", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.caption(f"Search: {config.EMBEDDING_BACKEND} embeddings + NumPy similarity · "
               f"Answers: Gemini ({config.GEMINI_CHAT_MODELS[0]}, with backups)")

# ---------- Header
st.title("🗳️ Johor State Election 2026")
st.caption(f"Answers come only from {n_articles} articles published by The Star from {first_date} to "
           f"{last_date} about the 16th Johor state election. Every answer cites its sources; questions outside "
           "this coverage are politely declined. Prototype: please check the linked articles for full details.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "\\$"))       # "$" would otherwise be read as maths
        if message.get("sources"):
            st.markdown("**Sources**\n" + "\n".join(
                f"- **[{s['number']}]** [{s['title']}]({s['url']}) ({s['date']})" for s in message["sources"]))
        if show_details and message.get("details"):
            st.caption(message["details"])


# ---------- Handle a new question (typed or example button)
question = st.chat_input("Ask about the 2026 Johor state election...") or clicked
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching The Star's articles..."):
        try:
            result = answer(question)
        except Exception as error:                       # e.g. no internet or Gemini unavailable
            result = {"answer": "Sorry, the answer service can't be reached right now. Please try again later.",
                      "sources": [], "reason": f"error: {str(error)[:150]}", "top_score": None, "model": None}
    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "sources": result["sources"],
        "details": f"Decision: {result['reason']} · best passage match score: {result['top_score']}"
                   + (f" · answered by: {result['model']}" if result.get("model") else "")})

if not st.session_state.messages:
    st.info("Ask a question below, or pick an example from the sidebar.")
for message in st.session_state.messages:
    show(message)
