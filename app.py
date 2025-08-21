import ast
import streamlit as st
from test_model_inference import ConversationalQueryRewriter

st.title("Conversational Query Rewriter")

@st.cache_resource
def load_model():
    return ConversationalQueryRewriter()

rewriter = load_model()

list_input_str = st.text_area(
    'Nhập list (ví dụ: ["a", "b", "c"])', value='["What is AI?", "AI is ...", "Tell me more"]'
)

try:
    list_input = ast.literal_eval(list_input_str)
    if not isinstance(list_input, list):
        st.error("Bạn phải nhập đúng định dạng list!")
        list_input = []
except Exception as e:
    st.error(f"Lỗi khi parse list: {e}")
    list_input = []

string_input = st.text_input("Nhập 1 chuỗi", value="What are examples?")

if st.button("Chạy"):
    if not list_input or not string_input.strip():
        st.warning("Bạn cần nhập cả lịch sử và truy vấn hiện tại.")
    else:
        with st.spinner("🔄 Đang xử lý..."):
            try:
                res = rewriter.generate_summary_query(list_input, string_input)
                if not res.strip():
                    st.warning("⚠️ Kết quả rỗng.")
                else:
                    st.success("✅ Output:")
                    st.code(res, language="text")
            except Exception as e:
                st.error(f"❌ Lỗi khi generate: {e}")
