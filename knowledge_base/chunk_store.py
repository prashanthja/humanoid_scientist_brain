# knowledge_base/chunk_store.py
# ------------------------------------------------------------
# ChunkStore — stores KB chunks for retrieval
# - Each chunk links to (paper_title, source)
# - Dedup by dedupe_key = sha1(normalized text)
# - SQLite only, includes auto-migration if schema is older
# ------------------------------------------------------------

from __future__ import annotations
import os
import sqlite3
import hashlib
from typing import List, Dict, Any


DB_PATH = "knowledge_base/knowledge.db"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _hash(text: str) -> str:
    t = _normalize_text(text)
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()


class ChunkStore:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path) or "knowledge_base", exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=30000;")
        self.cur = self.conn.cursor()
        self._create_or_migrate()

    def _table_cols(self, table: str) -> List[str]:
        self.cur.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in self.cur.fetchall()]

    def _create_or_migrate(self):
        # Base table
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                dedupe_key TEXT,
                paper_title TEXT DEFAULT 'unknown',
                source TEXT DEFAULT 'unknown',
                meta_json TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

        cols = set(self._table_cols("chunks"))

        if "dedupe_key" not in cols:
            self.cur.execute("ALTER TABLE chunks ADD COLUMN dedupe_key TEXT")
            self.conn.commit()

        # Backfill dedupe_key where missing/empty
        self.cur.execute("SELECT chunk_id, text, dedupe_key FROM chunks")
        rows = self.cur.fetchall()
        updated = 0
        for chunk_id, text, dk in rows:
            if dk and str(dk).strip():
                continue
            dk2 = _hash(text or "")
            try:
                self.cur.execute(
                    "UPDATE chunks SET dedupe_key = ? WHERE chunk_id = ?",
                    (dk2, int(chunk_id)),
                )
                updated += 1
            except Exception:
                # Ignore: duplicates will be handled below
                pass

        # Ensure UNIQUE index exists (this alone doesn't fix pre-existing dupes)
        self.cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_dedupe ON chunks(dedupe_key)")
        self.conn.commit()

        # If duplicates exist (same dedupe_key), keep earliest chunk_id and delete rest
        # This is important because backfill may create collisions.
        try:
            self.cur.execute("""
                DELETE FROM chunks
                WHERE chunk_id NOT IN (
                    SELECT MIN(chunk_id) FROM chunks
                    WHERE dedupe_key IS NOT NULL AND dedupe_key != ''
                    GROUP BY dedupe_key
                )
                AND dedupe_key IS NOT NULL AND dedupe_key != ''
            """)
            self.conn.commit()
        except Exception:
            pass

        if updated:
            print(f"🧩 ChunkStore migrated: backfilled dedupe_key for {updated} rows.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        added, skipped = 0, 0

        for c in chunks or []:
            raw_text = str(c.get("text") or "")
            text = _normalize_text(raw_text)
            if not text:
                skipped += 1
                continue

            # Quality gates: prevent garbage chunks (poisons embeddings)
            if len(text) < 160:  # slightly stricter than your 120
                skipped += 1
                continue

            title = str(c.get("paper_title") or "unknown").strip()
            # Drop "title-only" chunks (common poison)
            if title and title.lower() == text.lower():
                skipped += 1
                continue

            dk = _hash(text)
            source = str(c.get("source") or "unknown").strip()
            meta_json = str(c.get("meta_json") or "{}")

            try:
                self.cur.execute(
                    "INSERT INTO chunks (text, dedupe_key, paper_title, source, meta_json) VALUES (?, ?, ?, ?, ?)",
                    (text, dk, title or "unknown", source or "unknown", meta_json),
                )
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1

        self.conn.commit()
        return {"added": added, "skipped": skipped}

    def fetch_by_ids(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Fetch full chunks by chunk_id.
        Returns list[dict] with keys:
        chunk_id, text, paper_title, source, meta_json
        Preserves the order of chunk_ids (important for retrieval ranking).
        """
        if not chunk_ids:
            return []

        ids = [int(x) for x in chunk_ids]
        placeholders = ",".join(["?"] * len(ids))

        self.cur.execute(
            f"""
            SELECT chunk_id, text, paper_title, source, meta_json
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            ids,
        )
        rows = self.cur.fetchall()

        # Map for fast lookup
        by_id: Dict[int, Dict[str, Any]] = {}
        for chunk_id, text, paper_title, source, meta_json in rows:
            by_id[int(chunk_id)] = {
                "chunk_id": int(chunk_id),
                "text": str(text or ""),
                "paper_title": str(paper_title or "unknown"),
                "source": str(source or "unknown"),
                "meta_json": str(meta_json or "{}"),
            }

        # Return in the same order as requested ids
        out = []
        for cid in ids:
            if cid in by_id:
                out.append(by_id[cid])
        return out

    def fetch_recent(self, limit: int = 5000) -> List[Dict[str, Any]]:
        self.cur.execute(
            "SELECT chunk_id, text, paper_title, source, meta_json "
            "FROM chunks ORDER BY chunk_id DESC LIMIT ?",
            (int(limit),),
        )
        rows = self.cur.fetchall()
        out: List[Dict[str, Any]] = []
        for chunk_id, text, paper_title, source, meta_json in rows:
            out.append({
                "chunk_id": int(chunk_id),
                "text": str(text),
                "paper_title": str(paper_title),
                "source": str(source),
                "meta_json": str(meta_json),
            })
        return out[::-1]  # oldest -> newest

    def count(self) -> int:
        self.cur.execute("SELECT COUNT(*) FROM chunks")
        return int(self.cur.fetchone()[0])

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
