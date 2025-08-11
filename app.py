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

st.title("📚 知识库 + LLM Web Demo")
st.write("输入问题，我会从知识库检索信息并回答。")

query = st.text_input("请输入你的问题")
if st.button("提问") and query.strip():
    with st.spinner("正在检索并生成答案..."):
        answer = qa.run(query)
    st.markdown(f"**回答：** {answer}")