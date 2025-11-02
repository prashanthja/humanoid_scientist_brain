"""
Knowledge Base — Stores Text, Paper Metadata, and Source Traceability
--------------------------------------------------------------------
Now includes:
  • paper_title
  • concepts
  • source (e.g., 'internet', 'fetcher', etc.)
  • deduplication + null safety
"""

import sqlite3
import os
from typing import List, Dict

DB_PATH = "knowledge_base/knowledge.db"


class KnowledgeBase:
    def __init__(self):
        os.makedirs("knowledge_base", exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self._create_table()
        self.current_source = "unknown"

    # -------------------------------------------------------------
    # Schema creation
    # -------------------------------------------------------------
    def _create_table(self):
        self.cursor.execute("""
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

    # -------------------------------------------------------------
    # Store entries (robust against missing or duplicate data)
    # -------------------------------------------------------------
    def store(self, items: List[Dict]):
        """Store multiple entries (dicts or plain text)."""
        cur = self.conn.cursor()
        added, skipped = 0, 0

        for it in items:
            # Ensure valid data
            if isinstance(it, dict):
                text = str(it.get("text") or "").strip()
                title = str(it.get("paper_title") or "unknown").strip()
                concepts = str(it.get("concepts") or "unknown").strip()
                source = str(it.get("source") or self.current_source).strip()
            else:
                text = str(it or "").strip()
                title, concepts, source = "unknown", "unknown", self.current_source

            # Skip empty text or already-seen papers
            if not text:
                skipped += 1
                continue
            if self.exists(title):
                skipped += 1
                continue

            cur.execute(
                "INSERT INTO knowledge (text, paper_title, concepts, source) VALUES (?, ?, ?, ?)",
                (text, title, concepts, source),
            )
            added += 1

        self.conn.commit()
        print(f"📚 Stored {added} new items (skipped {skipped}) in KB (source={self.current_source})")

    # -------------------------------------------------------------
    # Query / existence / utilities
    # -------------------------------------------------------------
    def exists(self, title: str) -> bool:
        """Check if a paper title already exists in the database."""
        if not title or title.lower() in ["unknown", ""]:
            return False
        try:
            self.cursor.execute(
                "SELECT 1 FROM knowledge WHERE LOWER(paper_title) = ? LIMIT 1",
                (title.strip().lower(),),
            )
            return self.cursor.fetchone() is not None
        except Exception as e:
            print(f"⚠️ exists() failed for title '{title}': {e}")
            return False

    def query(self, keyword: str):
        """Return list of entries matching a keyword (case-insensitive)."""
        cur = self.conn.cursor()
        if not keyword:
            cur.execute("SELECT text, paper_title, concepts, source FROM knowledge")
        else:
            cur.execute(
                "SELECT text, paper_title, concepts, source FROM knowledge WHERE text LIKE ? OR paper_title LIKE ?",
                (f"%{keyword}%", f"%{keyword}%"),
            )
        rows = cur.fetchall()
        return [
            {"text": r[0], "paper_title": r[1], "concepts": r[2], "source": r[3]} for r in rows
        ]

    def fetch_all_with_embeddings(self):
        """Fetch all knowledge items along with metadata (for embedding rebuild or retraining)."""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT id, text, paper_title, concepts, source FROM knowledge")
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "text": r[1],
                    "paper_title": r[2],
                    "concepts": r[3],
                    "source": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch embeddings ({e}) — returning text only.")
            cur.execute("SELECT id, text FROM knowledge")
            rows = cur.fetchall()
            return [{"id": r[0], "text": r[1]} for r in rows]

    def count(self) -> int:
        """Return total count of entries in the knowledge base."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM knowledge")
        return cur.fetchone()[0]

    def close(self):
        """Close DB connection safely."""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
