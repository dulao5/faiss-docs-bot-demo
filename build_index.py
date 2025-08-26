# build_index.py
import os
import random
import sys
from collections import defaultdict
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def build_faiss_index(
    docs_dir: str = "docs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    save_path: str = "faiss_index"
) -> FAISS:
    """
    Build FAISS index from markdown documents
    """
    print("📂 Loading documents...")
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents found. Add .md files to {docs_dir} directory.")

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)

    # Add source information to each chunk
    for chunk in split_docs:
        src = chunk.metadata.get("source") or chunk.metadata.get("file_path") or "unknown"
        chunk.page_content = f"[Source: {src}]\n{chunk.page_content}"

    print(f"🛠 Vectorizing documents ({len(split_docs)} chunks)...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(save_path)
    print(f"✅ Vector index saved to {save_path}/")
    
    return db

def build_faq_samples(
    split_docs: List,
    faq_path: str = "docs/__faq_suggestions.txt",
    max_chunks: int = 10,
    max_chars: int = 3000,
    model: str = "gpt-4-0125-preview"
) -> None:
    """
    Generate FAQ samples from document chunks
    """
    print("🎯 Sampling from split_docs...")
    chunks_by_source = defaultdict(list)
    for chunk in split_docs:
        src = chunk.metadata.get("source") or chunk.metadata.get("file_path") or "unknown"
        chunks_by_source[src].append(chunk.page_content)

    all_parts = []
    for source, parts in chunks_by_source.items():
        random.shuffle(parts)
        for part in parts:
            all_parts.append({"page_content": part, "source": source})

    random.shuffle(all_parts)

    # Debug output
    sourcelist = set(part["source"] for part in all_parts)
    print(f"Extracted sources: {sourcelist}")

    sampled_parts = [part["page_content"] for part in all_parts[:max_chunks]]
    textlength = len("\n".join(sampled_parts))
    combined_text = "\n".join(sampled_parts)[:max_chars]
    print(f"✅ Sampling completed. Text length: {len(combined_text)} chars, limited by {textlength} chars")

    print("💡 Generating sample FAQ with LLM")
    llm = ChatOpenAI(model=model, temperature=0.3)

    prompt = f"""
    あなたはFAQの専門家です。以下のテキストから、ユーザーがよく尋ねる質問を10個生成してください。
    生成する質問は、以下の条件を満たしてください：
    1. 質問は簡潔で明確に。
    2. 各質問は異なるトピックをカバーする。
    3. 質問は箇条書き形式で、"- " または "• " で始める。
    4. 質問は日本語で書く。
    以下の形式で出力してください（質問のみの出力）：
    - 質問1
    - 質問2
    ...
    - 質問10
    --------------
    {combined_text}
    """

    resp = llm.invoke(prompt)
    questions_text = resp.content.strip()

    os.makedirs(os.path.dirname(faq_path), exist_ok=True)
    with open(faq_path, "w", encoding="utf-8") as f:
        for line in questions_text.splitlines():
            line = line.strip(" -•\t")
            if line:
                f.write(f"- {line}\n")

    print(f"✅ FAQ サンプルを {faq_path} に保存しました。")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 環境変数を設定してください")

    # Build FAISS index
    docs_dir = "docs"
    faiss_index_path = "faiss_index"
    faq_path = os.path.join(docs_dir, "__faq_suggestions.txt")

    db = build_faiss_index(
        docs_dir=docs_dir,
        save_path=faiss_index_path
    )

    # Generate FAQ samples using the existing documents
    build_faq_samples(
        split_docs=db.docstore._dict.values(),
        faq_path=faq_path
    )