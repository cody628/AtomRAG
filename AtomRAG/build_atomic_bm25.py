import json
import sqlite3
from tqdm import tqdm

JSON_PATH = "/workspace/AtomRAG/sillok/kv_store_text_atomics.json"
DB_PATH = "/workspace/AtomRAG/sillok/atomic_bm25.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS atomic_fts")
cur.execute("""
CREATE VIRTUAL TABLE atomic_fts USING fts5(
    atomic_id UNINDEXED,
    content,
    source_ids UNINDEXED
)
""")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for atomic_id, item in tqdm(data.items()):
    content = item.get("content", "")
    source_ids = json.dumps(item.get("source_ids", []), ensure_ascii=False)
    rows.append((atomic_id, content, source_ids))

cur.executemany(
    "INSERT INTO atomic_fts (atomic_id, content, source_ids) VALUES (?, ?, ?)",
    rows
)

conn.commit()
conn.close()