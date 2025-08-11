ローカルで簡単に動く RAG (Retrieval-Augmented Generation) Bot の Demo です。

# 0. ナレッジファイルを用意する
`docs/` に Markdown ファイル (`*.md`)を置く

# 1. OPENAI API key
`export OPENAI_API_KEY=Your OpenAI_API_KEY`

# 2. Docker 環境を作る
`docker compose build`

# 3. ナレッジベースをビルドする （初回実行時・ ナレッジを更新する時）
docker compose run --rm rag_app python build_index.py

# 4. webサービスを起動する
docker compose up