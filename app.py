import google.generativeai as genai
import streamlit as st

from rewrite import ConversationalQueryRewriter

# --- Cấu hình Gemini ---
genai.configure(api_key="AIzaSyBxOpnnneXjKtc4nF0LBFFGkyxY8OGS7Oo")
model = genai.GenerativeModel("gemini-1.5-flash")


@st.cache_resource
def load_model():
    return ConversationalQueryRewriter()


rewriter = load_model()
st.set_page_config(page_title="Chatbot Gemini", page_icon="🤖", layout="centered")

st.title("🤖 Chatbot với Gemini")

# --- Khởi tạo session state cho history ---
if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content), ...]

if "real" not in st.session_state:
    st.session_state.real = []  # [(role, content), ...]

if "rewrite_query" not in st.session_state:
    st.session_state.rewrite_query = ""


# --- Hàm gọi Gemini ---
def call_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text


# --- Hiển thị lịch sử chat ---
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)


# --- Nhập câu hỏi mới ---
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Lưu vào history
    st.session_state.history.append(("user", user_input))
    st.session_state.real.append(
        (
            "user",
            rewriter.rewrite(
                [c[1] for c in st.session_state.real], user_input, use_api=True
            ),
        )
    )

    # Gửi prompt đến Gemini
    full_context = "\n".join([f"{r}: {m}" for r, m in st.session_state.real])
    reply = call_gemini(full_context)

    # Lưu câu trả lời
    st.session_state.history.append(("assistant", reply))
    st.session_state.real.append(("assistant", reply))

    # Cập nhật lại màn hình
    st.rerun()


# --- Ô hiển thị câu query sau khi rewrite ---
st.subheader("📝 Query sau khi Rewrite")
if st.session_state.history:
    st.session_state.rewrite_query = st.session_state.real[-2][1] if len(st.session_state.real) > 1 else ""
st.text_area("Câu hỏi đã được rewrite:", st.session_state.rewrite_query, height=100)
