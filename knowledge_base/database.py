"""
Knowledge Base — Stores Text, Paper Metadata, and Source Traceability
--------------------------------------------------------------------
Includes:
  • paper_title
  • concepts
  • source
  • deduplication by text_hash (stable)
  • null safety
  • returns id in query() for embedding cache stability
"""

from __future__ import annotations

import os
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional

DB_PATH = "knowledge_base/knowledge.db"


class KnowledgeBase:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path) or "knowledge_base", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # more robust for frequent writes
        self.cursor = self.conn.cursor()
        self._create_or_migrate()
        self.current_source = "unknown"

    # -------------------------------------------------------------
    # Schema creation + migration
    # -------------------------------------------------------------
    def _create_or_migrate(self):
        # Create base table if missing (includes text_hash)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                text_hash TEXT,
                paper_title TEXT DEFAULT 'unknown',
                concepts TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'unknown',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        # Migrate older DBs that don't have text_hash
        try:
            cols = {row[1] for row in self.cursor.execute("PRAGMA table_info(knowledge)").fetchall()}
            if "text_hash" not in cols:
                self.cursor.execute("ALTER TABLE knowledge ADD COLUMN text_hash TEXT")
                self.conn.commit()
        except Exception:
            pass

        # Unique index for deduplication
        try:
            self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_text_hash ON knowledge(text_hash)")
            self.conn.commit()
        except Exception:
            pass

        # Helpful indices
        try:
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge(paper_title)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge(source)")
            self.conn.commit()
        except Exception:
            pass

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _clean(self, x: Any, default: str = "") -> str:
        s = str(x) if x is not None else ""
        s = s.strip()
        return s if s else default

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    # -------------------------------------------------------------
    # Store entries (dedup + null safe)
    # -------------------------------------------------------------
    def store(self, items: List[Any]) -> None:
        """
        Store multiple entries (dicts or plain text).
        Dedup is by text_hash (stable).
        """
        cur = self.conn.cursor()
        added, skipped = 0, 0

        for it in items or []:
            if isinstance(it, dict):
                text = self._clean(it.get("text"), default="")
                title = self._clean(it.get("paper_title"), default="unknown")
                concepts = self._clean(it.get("concepts"), default="unknown")
                source = self._clean(it.get("source"), default=self.current_source or "unknown")
            else:
                text = self._clean(it, default="")
                title, concepts, source = "unknown", "unknown", (self.current_source or "unknown")

            if not text:
                skipped += 1
                continue

            text_hash = self._hash_text(text)

            # Insert with UNIQUE(text_hash). If already exists -> skip.
            try:
                cur.execute(
                    "INSERT INTO knowledge (text, text_hash, paper_title, concepts, source) VALUES (?, ?, ?, ?, ?)",
                    (text, text_hash, title, concepts, source),
                )
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
                continue

        self.conn.commit()
        print(f"📚 Stored {added} new items (skipped {skipped}) in KB (source={self.current_source})")

    # -------------------------------------------------------------
    # Query / utilities
    # -------------------------------------------------------------
    def query(self, keyword: str):
        """
        Return list of entries matching keyword (case-insensitive).
        IMPORTANT: includes id + text_hash for stable downstream embedding caches.
        """
        cur = self.conn.cursor()
        kw = self._clean(keyword, default="")

        if not kw:
            cur.execute("SELECT id, text, text_hash, paper_title, concepts, source FROM knowledge")
        else:
            like = f"%{kw}%"
            cur.execute(
                "SELECT id, text, text_hash, paper_title, concepts, source "
                "FROM knowledge WHERE text LIKE ? OR paper_title LIKE ? OR concepts LIKE ?",
                (like, like, like),
            )

        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "text_hash": r[2],
                "paper_title": r[3],
                "concepts": r[4],
                "source": r[5],
            }
            for r in rows
        ]

    def fetch_all_with_embeddings(self):
        """
        Fetch all items with metadata.
        (This name is legacy; it fetches rows used for embedding rebuild.)
        """
        return self.query("")

    def count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM knowledge")
        return int(cur.fetchone()[0])

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
