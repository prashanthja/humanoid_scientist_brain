# knowledge_base/storage.py
import sqlite3
import time
import numpy as np
import io
from typing import List, Tuple, Optional

DB_PATH = "knowledge_base/knowledge.db"


def _to_blob(np_array: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for SQLite BLOB."""
    memfile = io.BytesIO()
    np.save(memfile, np_array)
    memfile.seek(0)
    return memfile.read()


def _from_blob(blob: bytes) -> np.ndarray:
    memfile = io.BytesIO(blob)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


class KnowledgeStorage:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            timestamp REAL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY(id) REFERENCES knowledge(id) ON DELETE CASCADE
        );
        """)
        self.conn.commit()

    def store(self, content: str, source: Optional[str] = None, embedding: Optional[np.ndarray] = None) -> int:
        """Store content and optional embedding. Returns inserted knowledge id."""
        cur = self.conn.cursor()
        ts = time.time()
        cur.execute(
            "INSERT INTO knowledge (content, source, timestamp) VALUES (?, ?, ?)",
            (content, source, ts),
        )
        k_id = cur.lastrowid
        if embedding is not None:
            blob = _to_blob(np.asarray(embedding, dtype=np.float32))
            cur.execute("INSERT INTO embeddings (id, vector) VALUES (?, ?)", (k_id, blob))
        self.conn.commit()
        return k_id

    def update_embedding(self, k_id: int, embedding: np.ndarray):
        cur = self.conn.cursor()
        blob = _to_blob(np.asarray(embedding, dtype=np.float32))
        cur.execute("REPLACE INTO embeddings (id, vector) VALUES (?, ?)", (k_id, blob))
        self.conn.commit()

    def bulk_store(self, items: List[Tuple[str, Optional[str], Optional[np.ndarray]]]) -> List[int]:
        """
        items: list of (content, source, embedding) where embedding may be None.
        Returns list of inserted ids.
        """
        ids = []
        for content, source, embedding in items:
            ids.append(self.store(content, source, embedding))
        return ids

    def fetch_all_with_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Return list of (id, content, vector) for all records that have embeddings."""
        cur = self.conn.cursor()
        cur.execute("""
        SELECT k.id, k.content, e.vector
        FROM knowledge k
        JOIN embeddings e ON k.id = e.id
        """)
        rows = cur.fetchall()
        out: List[Tuple[int, str, np.ndarray]] = []
        for k_id, content, blob in rows:
            out.append((int(k_id), str(content), _from_blob(blob)))
        return out

    def fetch_all_contents(self) -> List[Tuple[int, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM knowledge")
        rows = cur.fetchall()
        return [(int(i), str(c)) for i, c in rows]

    def query_keyword(self, keyword: str, limit: int = 10) -> List[Tuple[int, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, content FROM knowledge WHERE content LIKE ? LIMIT ?",
            (f"%{keyword}%", int(limit)),
        )
        rows = cur.fetchall()
        return [(int(i), str(c)) for i, c in rows]

    def get(self, k_id: int) -> Tuple[int, str]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM knowledge WHERE id = ?", (int(k_id),))
        row = cur.fetchone()
        if not row:
            return (-1, "")
        return (int(row[0]), str(row[1]))

    def close(self):
        self.conn.close()
