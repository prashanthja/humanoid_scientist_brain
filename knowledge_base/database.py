import sqlite3
import time
import numpy as np
import io

def _to_blob(np_array: np.ndarray) -> bytes:
    memfile = io.BytesIO()
    np.save(memfile, np_array)
    memfile.seek(0)
    return memfile.read()

def _from_blob(blob: bytes) -> np.ndarray:
    memfile = io.BytesIO(blob)
    memfile.seek(0)
    return np.load(memfile)

class KnowledgeBase:
    def __init__(self, db_path="knowledge_base/knowledge.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.create_table()
        self.create_embedding_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp REAL
        )
        """)
        self.conn.commit()

    def create_embedding_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            FOREIGN KEY(id) REFERENCES knowledge(id) ON DELETE CASCADE
        )
        """)
        self.conn.commit()

    def store(self, data_list, embeddings=None):
        """
        Store data_list (list of texts). Optionally store embeddings list of np.ndarray in parallel.
        """
        cursor = self.conn.cursor()
        count = 0
        for idx, text in enumerate(data_list):
            cursor.execute(
                "INSERT INTO knowledge (content, timestamp) VALUES (?, ?)",
                (text, time.time())
            )
            k_id = cursor.lastrowid
            if embeddings is not None and len(embeddings) > idx and embeddings[idx] is not None:
                blob = _to_blob(embeddings[idx].astype(np.float32))
                cursor.execute(
                    "INSERT INTO embeddings (id, vector) VALUES (?, ?)",
                    (k_id, blob)
                )
            count += 1
        self.conn.commit()
        print(f"📚 Stored {count} new items in KB")

    def query(self, keyword):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT content FROM knowledge WHERE content LIKE ?",
            (f"%{keyword}%",)
        )
        results = [row[0] for row in cursor.fetchall()]
        return results

    def fetch_all_with_embeddings(self):
        """
        Return list of (id, content, vector) for all knowledge entries that have embeddings.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT k.id, k.content, e.vector
        FROM knowledge k
        JOIN embeddings e ON k.id = e.id
        """)
        rows = cursor.fetchall()
        results = []
        for k_id, content, blob in rows:
            vec = _from_blob(blob)
            results.append((k_id, content, vec))
        return results

    def close(self):
        self.conn.close()
