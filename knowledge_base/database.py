# knowledge_base/database.py
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
from typing import List, Dict, Any

DB_PATH = "knowledge_base/knowledge.db"


class KnowledgeBase:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path) or "knowledge_base", exist_ok=True)

        # timeout + check_same_thread helps with frequent writes + multiple modules/threads
        self.conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")        # better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL;")      # WAL-friendly
        self.cursor = self.conn.cursor()

        self._create_or_migrate()
        self.current_source = "unknown"

    # -------------------------------------------------------------
    # Schema creation + migration
    # -------------------------------------------------------------
    def _create_or_migrate(self):
        # Create table with modern schema
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                text_hash TEXT,
                paper_title TEXT DEFAULT 'unknown',
                concepts TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'unknown',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        # Detect existing columns
        try:
            cols = {row[1] for row in self.cursor.execute("PRAGMA table_info(knowledge)").fetchall()}
        except Exception:
            cols = set()

        # Migrate missing columns (older DBs)
        def _add_col(name: str, ddl: str):
            if name not in cols:
                try:
                    self.cursor.execute(ddl)
                    self.conn.commit()
                except Exception:
                    pass

        _add_col("text_hash", "ALTER TABLE knowledge ADD COLUMN text_hash TEXT")
        _add_col("paper_title", "ALTER TABLE knowledge ADD COLUMN paper_title TEXT DEFAULT 'unknown'")
        _add_col("concepts", "ALTER TABLE knowledge ADD COLUMN concepts TEXT DEFAULT 'unknown'")
        _add_col("source", "ALTER TABLE knowledge ADD COLUMN source TEXT DEFAULT 'unknown'")
        _add_col("timestamp", "ALTER TABLE knowledge ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")

        # Indices
        try:
            self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_text_hash ON knowledge(text_hash)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge(paper_title)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge(source)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_timestamp ON knowledge(timestamp)")
            self.conn.commit()
        except Exception:
            pass

        # -----------------------------
        # Backfill + dedupe old rows
        # -----------------------------
        try:
            # Backfill missing/empty text_hash
            self.cursor.execute("SELECT id, text, text_hash FROM knowledge")
            rows = self.cursor.fetchall()
            updated = 0
            for _id, text, th in rows:
                if th and str(th).strip():
                    continue
                t = self._clean(text, default="")
                if not t:
                    continue
                th2 = self._hash_text(t)
                try:
                    self.cursor.execute(
                        "UPDATE knowledge SET text_hash = ? WHERE id = ?",
                        (th2, int(_id)),
                    )
                    updated += 1
                except Exception:
                    pass

            # Delete duplicates (keep earliest id per text_hash)
            self.cursor.execute("""
                DELETE FROM knowledge
                WHERE id NOT IN (
                    SELECT MIN(id) FROM knowledge
                    WHERE text_hash IS NOT NULL AND text_hash != ''
                    GROUP BY text_hash
                )
                AND text_hash IS NOT NULL AND text_hash != ''
            """)

            self.conn.commit()
            if updated:
                print(f"📌 KB migrated: backfilled text_hash for {updated} rows and deduped.")
        except Exception:
            pass

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def set_source(self, source: str):
        self.current_source = self._clean(source, default="unknown") or "unknown"

    def _clean(self, x: Any, default: str = "") -> str:
        s = str(x) if x is not None else ""
        s = " ".join(s.replace("\u00a0", " ").split()).strip()
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
                    "INSERT INTO knowledge (text, text_hash, paper_title, concepts, source) "
                    "VALUES (?, ?, ?, ?, ?)",
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

    def query(self, keyword: str, *, limit: int = 2000):
        """
        Return list of entries matching keyword (case-insensitive).
        - Deterministic ordering (newest first)
        - Filters out broken rows (missing text_hash)
        - Bounded (limit)
        """
        cur = self.conn.cursor()
        kw = self._clean(keyword, default="")

        # Always ignore rows with missing hash (these are usually legacy dupes)
        base_where = "WHERE text_hash IS NOT NULL AND text_hash != ''"

        if not kw:
            cur.execute(
                "SELECT id, text, text_hash, paper_title, concepts, source "
                "FROM knowledge "
                f"{base_where} "
                "ORDER BY timestamp DESC, id DESC "
                "LIMIT ?",
                (int(limit),),
            )
        else:
            like = f"%{kw}%"
            cur.execute(
                "SELECT id, text, text_hash, paper_title, concepts, source "
                "FROM knowledge "
                f"{base_where} AND (text LIKE ? OR paper_title LIKE ? OR concepts LIKE ?) "
                "ORDER BY timestamp DESC, id DESC "
                "LIMIT ?",
                (like, like, like, int(limit)),
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
        Legacy name: used by embedding rebuild logic.
        Returns rows with stable ids.
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
