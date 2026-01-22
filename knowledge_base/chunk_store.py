# knowledge_base/chunk_store.py
# ------------------------------------------------------------
# ChunkStore — stores KB chunks for retrieval (the real fix)
# - Each chunk links to (paper_title, source)
# - Dedup by hash(text)
# - SQLite only
# ------------------------------------------------------------

from __future__ import annotations
import os, sqlite3, hashlib
from typing import List, Dict, Any, Optional

DB_PATH = "knowledge_base/knowledge.db"


def _hash(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8", errors="ignore")).hexdigest()


class ChunkStore:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create()

    def _create(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                text_hash TEXT UNIQUE,
                paper_title TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'unknown',
                meta_json TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        added, skipped = 0, 0
        for c in chunks:
            text = str(c.get("text") or "").strip()
            if not text:
                skipped += 1
                continue
            h = _hash(text)
            paper_title = str(c.get("paper_title") or "unknown")
            source = str(c.get("source") or "unknown")
            meta_json = str(c.get("meta_json") or "{}")

            try:
                self.cur.execute(
                    "INSERT INTO chunks (text, text_hash, paper_title, source, meta_json) VALUES (?, ?, ?, ?, ?)",
                    (text, h, paper_title, source, meta_json)
                )
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
                continue

        self.conn.commit()
        return {"added": added, "skipped": skipped}

    def fetch_recent(self, limit: int = 5000) -> List[Dict[str, Any]]:
        self.cur.execute(
            "SELECT chunk_id, text, paper_title, source, meta_json FROM chunks ORDER BY chunk_id DESC LIMIT ?",
            (int(limit),)
        )
        rows = self.cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "chunk_id": r[0],
                "text": r[1],
                "paper_title": r[2],
                "source": r[3],
                "meta_json": r[4],
            })
        return out[::-1]  # oldest->newest

    def count(self) -> int:
        self.cur.execute("SELECT COUNT(*) FROM chunks")
        return int(self.cur.fetchone()[0])

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
