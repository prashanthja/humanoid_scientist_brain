"""
Knowledge Base — Stores Text, Paper Metadata, and Source Traceability
--------------------------------------------------------------------
Now includes columns for:
- paper_title
- concepts
- source (e.g., 'internet', 'fetcher', etc.)
"""

import sqlite3, os, json

DB_PATH = "knowledge_base/knowledge.db"


class KnowledgeBase:
    def __init__(self):
        os.makedirs("knowledge_base", exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self._create_table()
        self.current_source = "unknown"

    def _create_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                paper_title TEXT DEFAULT 'unknown',
                concepts TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'unknown',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def store(self, items):
        """Store multiple knowledge entries (supports dict or raw text)."""
        cur = self.conn.cursor()
        for it in items:
            if isinstance(it, dict):
                text = it.get("text", "")
                title = it.get("paper_title", "unknown")
                concepts = it.get("concepts", "unknown")
                source = it.get("source", self.current_source)
            else:
                text, title, concepts, source = str(it), "unknown", "unknown", self.current_source

            cur.execute(
                "INSERT INTO knowledge (text, paper_title, concepts, source) VALUES (?, ?, ?, ?)",
                (text, title, concepts, source),
            )
        self.conn.commit()
        print(f"📚 Stored {len(items)} new items in KB (source={self.current_source})")

    def query(self, keyword):
        """Return list of entries matching keyword."""
        cur = self.conn.cursor()
        if not keyword:
            cur.execute("SELECT text, paper_title, concepts, source FROM knowledge")
        else:
            cur.execute(
                "SELECT text, paper_title, concepts, source FROM knowledge WHERE text LIKE ?",
                (f"%{keyword}%",),
            )
        rows = cur.fetchall()
        return [
            {"text": r[0], "paper_title": r[1], "concepts": r[2], "source": r[3]} for r in rows
        ]
    
    def fetch_all_with_embeddings(self):
        """
        Fetch all knowledge items along with their embeddings (if stored).
        If no embeddings column exists yet, it returns text only.
        """
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT id, text, paper_title, concepts, source FROM knowledge")
            rows = cur.fetchall()
            result = []
            for r in rows:
                result.append({
                    "id": r[0],
                    "text": r[1],
                    "paper_title": r[2],
                    "concepts": r[3],
                    "source": r[4],
                })
            return result
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch embeddings ({e}) — returning text only.")
            cur.execute("SELECT id, text FROM knowledge")
            rows = cur.fetchall()
            return [{"id": r[0], "text": r[1]} for r in rows]

    def close(self):
        self.conn.close()
    