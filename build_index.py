# build_index.py
import os
import random
from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ------- 配置 -------
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY 環境変数を設定してください")

DOCS_DIR = "docs"
FAQ_PATH = os.path.join(DOCS_DIR, "__faq_suggestions.txt")

# サンプリング設定
PER_SOURCE_MAX = 5     # ソースごとに抽出する最大数
MAX_CHUNKS = 10        # サンプリングの最大チャンク数
MAX_CHARS = 3000       # 生成するテキストの最大文字数
# ---------------------

print("📂 ドキュメントをロードする...")
loader = DirectoryLoader(
    DOCS_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
if not docs:
    raise ValueError("ドキュメントが見つかりません。docs ディレクトリに .md ファイルを追加してください。")

print("✂️ Chunkで分割...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

print(f"🛠 ベクタードキュメント（{len(split_docs)} chunks）...")
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)
db.save_local("faiss_index")
print("✅ ベクターインデックスを faiss_index/ に保存しました。")

# ----------------- FAQ サンプル生成 -----------------
print("🎯 split_docs からサンプリング...")
chunks_by_source = defaultdict(list)
for chunk in split_docs:
    src = chunk.metadata.get("source") or chunk.metadata.get("file_path") or "unknown"
    chunks_by_source[src].append(chunk.page_content)

sampled_parts = []
all_parts = [] # item is {"page_content": str, "source": str}

for source, parts in chunks_by_source.items():
    random.shuffle(parts)
    for part in parts:
        all_parts.append({"page_content": part, "source": source})

random.shuffle(all_parts)

# random の効果をDebug
# source を出力
sourcelist = [part["source"] for part in all_parts]
sourcelist = set(sourcelist)
print(f"抽出されたソース: {sourcelist}")

# from all_parts[:MAX_CHUNKS]
sampled_parts = [part["page_content"] for part in all_parts[:MAX_CHUNKS]]

textlength = len("\n".join(sampled_parts))
combined_text = "\n".join(sampled_parts)[:MAX_CHARS]
print(f"✅ サンプリング完了。抽出されたテキストの長さ: {len(combined_text)} 文字 , limited by {textlength} 文字")

# ----------------- FAQ サンプル生成 -----------------
print("💡 LLM で サンプル FAQ を生成")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

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

# ----------------- FAQ サンプル保存 -----------------
with open(FAQ_PATH, "w", encoding="utf-8") as f:
    for line in questions_text.splitlines():
        line = line.strip(" -•\t")
        if line:
            f.write(f"- {line}\n")

print(f"✅ FAQ サンプルを {FAQ_PATH} に保存しました。")