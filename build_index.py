# build_index.py
import os
import random
from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
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

# 1) 各ソースからランダムに抽出
for src, parts in chunks_by_source.items():
    n = min(len(parts), PER_SOURCE_MAX)
    sampled = random.sample(parts, n)
    sampled_parts.extend(sampled)
    if len(sampled_parts) >= MAX_CHUNKS:
        break

# 2) もしまだ足りない場合、全体からランダムに抽出
if len(sampled_parts) < MAX_CHUNKS:
    all_parts = [c.page_content for c in split_docs]
    # 除外済みの部分を除く
    remaining = [p for p in all_parts if p not in sampled_parts]
    need = min(len(remaining), MAX_CHUNKS - len(sampled_parts))
    if need > 0:
        sampled_parts.extend(random.sample(remaining, need))

# 3) 抽出した部分をランダムに並べ替え、最大文字数に制限
random.shuffle(sampled_parts)
combined_text = "\n".join(sampled_parts)[:MAX_CHARS]

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