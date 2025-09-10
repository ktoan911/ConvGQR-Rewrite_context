import streamlit as st

from rewrite_local import Rewrite
from utils import create_gemini_client

model = create_gemini_client("gemini-1.5-flash")

@st.cache_resource
def load_model():
    return Rewrite()


rewriter = load_model()

st.set_page_config(page_title="Chatbot Gemini", page_icon="🤖", layout="centered")

st.title("🤖 Chatbot với Gemini")

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content), ...]

if "real" not in st.session_state:
    st.session_state.real = []

if "rewrite_query" not in st.session_state:
    st.session_state.rewrite_query = ""


def call_gemini(history):
    context = "\n".join([f"{r}: {m}" for r, m in history[-20:]])
    response = model(context)
    return response


for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)


user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.history.append(("user", user_input))

    rewritten = rewriter.rewrite_query([c[1] for c in st.session_state.history], user_input)
    st.session_state.real.append(("user", rewritten))

    reply = call_gemini(st.session_state.history)

    st.session_state.history.append(("assistant", reply))
    st.session_state.real.append(("assistant", reply))

    st.rerun()


st.subheader("📝 Query sau khi Rewrite")
if len(st.session_state.real) > 1:
    st.session_state.rewrite_query = st.session_state.real[-2][1]  # câu user đã rewrite

st.text_area(
    "Câu hỏi đã được rewrite:",
    st.session_state.rewrite_query,
    height=100,
    disabled=True,
)
