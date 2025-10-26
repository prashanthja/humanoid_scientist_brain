# scripts/migrate_kb.py
import sqlite3, json, os

DB = "knowledge_base/knowledge.db"

def column_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def table_exists(cur, table):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Ensure main table exists
cur.execute("""
CREATE TABLE IF NOT EXISTS knowledge (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  timestamp REAL
)
""")

# Add traceability columns (SQLite doesn't support IF NOT EXISTS on ADD COLUMN)
for col, sqltype in [
    ("source", "TEXT"),
    ("title", "TEXT"),
    ("url", "TEXT"),
    ("tags", "TEXT"),      # JSON-encoded list
    ("concepts", "TEXT")   # JSON-encoded list you derive later
]:
    if not column_exists(cur, "knowledge", col):
        cur.execute(f"ALTER TABLE knowledge ADD COLUMN {col} {sqltype}")

# Optional embeddings table (if you use it)
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
  id INTEGER PRIMARY KEY,
  vector BLOB NOT NULL,
  FOREIGN KEY(id) REFERENCES knowledge(id) ON DELETE CASCADE
)
""")

# Learning log (what concepts were learned from which item)
if not table_exists(cur, "learning_log"):
    cur.execute("""
    CREATE TABLE learning_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      knowledge_id INTEGER,
      learned_concepts TEXT, -- JSON list
      created_at REAL,
      FOREIGN KEY(knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
    )
    """)

conn.commit()
conn.close()
print("✅ migration complete.")
