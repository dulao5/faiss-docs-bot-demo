import os

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set system variable OPENAI_API_KEY")


loader = DirectoryLoader(
    "docs",
    glob="**/*.md",
    loader_cls=TextLoader,     # 指定用 TextLoader
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)

db.save_local("faiss_index")

print("✅ builed vector index")