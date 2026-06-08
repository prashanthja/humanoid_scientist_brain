"""
Private Corpus Manager
----------------------
Each enterprise client gets their own Pinecone namespace.
Their documents never mix with public corpus or other clients.
"""

import os, sqlite3, hashlib, json, time
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "knowledge_base/knowledge.db"
CHUNK_SIZE = 300  # tokens approx
CHUNK_OVERLAP = 50


def init_private_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS private_companies (
            company_id   TEXT PRIMARY KEY,
            company_name TEXT,
            namespace    TEXT UNIQUE,
            password     TEXT,
            created_at   TEXT,
            chunk_count  INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS private_chunks (
            chunk_id     TEXT PRIMARY KEY,
            company_id   TEXT,
            text         TEXT,
            paper_title  TEXT,
            source       TEXT,
            filename     TEXT,
            created_at   TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_company(company_name: str, password: str, domain: str = "ml") -> Dict:
    """Create a new enterprise client namespace."""
    company_id = hashlib.md5(company_name.lower().encode()).hexdigest()[:10]
    namespace = f"company-{company_id}"
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT OR IGNORE INTO private_companies
            (company_id, company_name, namespace, password, created_at, chunk_count, domain)
            VALUES (?,?,?,?,?,0,?)
        """, (company_id, company_name, namespace, password, datetime.now().isoformat(), domain))
        conn.commit()
    finally:
        conn.close()
    return {"company_id": company_id, "namespace": namespace}


def get_company(company_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT company_id, company_name, namespace, chunk_count FROM private_companies WHERE company_id=?",
        (company_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"company_id": row[0], "company_name": row[1], "namespace": row[2], "chunk_count": row[3]}


def verify_company(company_id: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT password FROM private_companies WHERE company_id=?", (company_id,)
    ).fetchone()
    conn.close()
    return row and row[0] == password


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages[:20]:  # max 20 pages
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip()
    except Exception as e:
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages[:20]:
                text += page.extract_text() or ""
            return text.strip()
        except:
            return ""


def chunk_text(text: str, title: str = "") -> List[Dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text) > 50:
            chunk_id = hashlib.md5(chunk_text[:100].encode()).hexdigest()[:12]
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "paper_title": title,
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_document(company_id: str, pdf_bytes: bytes,
                    filename: str, title: str = "") -> Dict:
    """
    Ingest a PDF into company's private namespace.
    Returns: {chunks_added, namespace, status}
    """
    init_private_db()

    company = get_company(company_id)
    if not company:
        return {"error": "Company not found"}

    namespace = company["namespace"]
    if not title:
        title = filename.replace(".pdf", "").replace("_", " ")

    # Extract text
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        return {"error": "Could not extract text from PDF"}

    # Chunk
    chunks = chunk_text(text, title)
    if not chunks:
        return {"error": "No content extracted"}

    # Embed + upsert to private Pinecone namespace
    try:
        import sys; sys.path.insert(0, '.')
        from dotenv import load_dotenv; load_dotenv('.env')
        from learning_module.embedding_bridge import EmbeddingBridge
        from pinecone import Pinecone
        import numpy as np

        encoder = EmbeddingBridge()
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        idx = pc.Index(os.environ['PINECONE_INDEX'])

        texts = [c["text"] for c in chunks]
        embeddings = encoder.embed(texts)

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{company_id}-{chunk['chunk_id']}",
                "values": emb.tolist(),
                "metadata": {
                    "text": chunk["text"][:500],
                    "paper_title": title,
                    "source": "private",
                    "company_id": company_id,
                    "filename": filename,
                }
            })

        # Upsert in batches of 100
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            idx.upsert(vectors=batch, namespace=namespace)

        # Save to SQLite
        conn = sqlite3.connect(DB_PATH)
        now = datetime.now().isoformat()
        for chunk in chunks:
            conn.execute("""
                INSERT OR IGNORE INTO private_chunks
                (chunk_id, company_id, text, paper_title, source, filename, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (chunk["chunk_id"], company_id, chunk["text"][:1000],
                  title, "private", filename, now))
        conn.execute("""
            UPDATE private_companies
            SET chunk_count = chunk_count + ?
            WHERE company_id = ?
        """, (len(chunks), company_id))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "chunks_added": len(chunks),
            "namespace": namespace,
            "title": title,
            "filename": filename,
        }

    except Exception as e:
        return {"error": str(e)}


def search_private(company_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """Search company's private namespace."""
    company = get_company(company_id)
    if not company:
        return []

    try:
        import sys; sys.path.insert(0, '.')
        from dotenv import load_dotenv; load_dotenv('.env')
        from learning_module.embedding_bridge import EmbeddingBridge
        from pinecone import Pinecone

        encoder = EmbeddingBridge()
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        idx = pc.Index(os.environ['PINECONE_INDEX'])

        query_emb = encoder.embed([query])[0].tolist()
        results = idx.query(
            vector=query_emb,
            top_k=top_k,
            namespace=company["namespace"],
            include_metadata=True
        )

        chunks = []
        for match in results.matches:
            meta = match.metadata or {}
            chunks.append({
                "text": meta.get("text", ""),
                "paper_title": meta.get("paper_title", ""),
                "source": "private",
                "score": match.score,
                "filename": meta.get("filename", ""),
            })
        return chunks

    except Exception as e:
        print(f"Private search error: {e}")
        return []


def get_company_stats(company_id: str) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    company = conn.execute(
        "SELECT company_name, namespace, chunk_count, created_at FROM private_companies WHERE company_id=?",
        (company_id,)
    ).fetchone()
    if not company:
        conn.close()
        return {}
    docs = conn.execute(
        "SELECT DISTINCT filename, paper_title, COUNT(*) as chunks FROM private_chunks WHERE company_id=? GROUP BY filename",
        (company_id,)
    ).fetchall()
    conn.close()
    return {
        "company_name": company[0],
        "namespace": company[1],
        "total_chunks": company[2],
        "created_at": company[3],
        "documents": [{"filename": d[0], "title": d[1], "chunks": d[2]} for d in docs]
    }


if __name__ == "__main__":
    init_private_db()
    # Test
    c = create_company("Acme AI Research", "test-password-123")
    print("Created:", c)
    stats = get_company_stats(c["company_id"])
    print("Stats:", stats)
