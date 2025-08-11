import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please she the system variable OPENAI_API_KEY")
    st.stop()

@st.cache_resource
def load_qa():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

qa = load_qa()

st.set_page_config(page_title="ナレッジベースボット Demo", layout="wide")
st.title("📚 ナレッジベースボット Demo")

faq_file = "docs/__faq_suggestions.txt"
if os.path.exists(faq_file):
    st.subheader("💡 FAQ サンプル")
    with open(faq_file, "r", encoding="utf-8") as f:
        faq_lines = [line.strip(" -•\t") for line in f if line.strip()]

    col1, col2 = st.columns(2)
    for i, q in enumerate(faq_lines):
        if i % 2 == 0:
            with col1:
                if st.button(q, key=f"faq_{i}"):
                    st.session_state.query = q
        else:
            with col2:
                if st.button(q, key=f"faq_{i}"):
                    st.session_state.query = q

st.subheader("❓ 質問を入力")

if "query" not in st.session_state:
    st.session_state.query = ""

with st.form("query_form", clear_on_submit=False):
    query = st.text_input(
        "質問を入力してください",
        placeholder="面白いことを教えてください",
        key="query"
    )
    submitted = st.form_submit_button("Post")
    if submitted and query.strip():
        with st.spinner("回答を生成中..."):
            answer = qa.run(query)
        st.markdown(f"**回答：** {answer}")