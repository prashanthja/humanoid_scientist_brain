"""
Private Corpus Manager — Supabase backend
Each enterprise client gets their own Pinecone namespace.
Uses Supabase for persistence (works on Render).
"""

import os, hashlib, json, time
from datetime import datetime
from typing import List, Dict, Optional


def _sb():
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv('.env')
    return create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])


def create_company(company_name: str, password: str, domain: str = "ml") -> Dict:
    company_id = hashlib.md5(company_name.lower().encode()).hexdigest()[:10]
    namespace  = f"company-{company_id}"
    try:
        _sb().table("private_companies").upsert({
            "company_id":   company_id,
            "company_name": company_name,
            "namespace":    namespace,
            "password":     password,
            "domain":       domain,
            "created_at":   datetime.now().isoformat(),
            "chunk_count":  0,
        }).execute()
    except Exception as e:
        print(f"create_company error: {e}")
    return {"company_id": company_id, "namespace": namespace}


def get_company(company_id: str) -> Optional[Dict]:
    try:
        r = _sb().table("private_companies").select("*").eq("company_id", company_id).execute()
        if r.data:
            return r.data[0]
    except Exception as e:
        print(f"get_company error: {e}")
    return None


def verify_company(company_id: str, password: str) -> bool:
    c = get_company(company_id)
    return bool(c and c.get("password") == password)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages[:20]:
                t = page.extract_text()
                if t: text += t + "\n"
        return text.strip()
    except:
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            return "".join(p.extract_text() or "" for p in reader.pages[:20]).strip()
        except:
            return ""


def chunk_text(text: str, title: str = "") -> List[Dict]:
    CHUNK_SIZE, OVERLAP = 300, 50
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk_words = words[i:i+CHUNK_SIZE]
        ct = " ".join(chunk_words)
        if len(ct) > 50:
            cid = hashlib.md5(ct[:100].encode()).hexdigest()[:12]
            chunks.append({"chunk_id": cid, "text": ct, "paper_title": title})
        i += CHUNK_SIZE - OVERLAP
    return chunks


def ingest_document(company_id: str, pdf_bytes: bytes,
                    filename: str, title: str = "") -> Dict:
    company = get_company(company_id)
    if not company:
        return {"error": "Company not found"}
    namespace = company["namespace"]
    if not title:
        title = filename.replace(".pdf","").replace("_"," ")

    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        return {"error": "Could not extract text from PDF"}

    chunks = chunk_text(text, title)
    if not chunks:
        return {"error": "No content extracted"}

    try:
        import sys; sys.path.insert(0,'.')
        from dotenv import load_dotenv; load_dotenv('.env')
        from learning_module.embedding_bridge import EmbeddingBridge
        from pinecone import Pinecone

        encoder = EmbeddingBridge()
        pc  = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        idx = pc.Index(os.environ['PINECONE_INDEX'])

        texts      = [c["text"] for c in chunks]
        embeddings = encoder.embed(texts)

        vectors = [{
            "id": f"{company_id}-{c['chunk_id']}",
            "values": e.tolist(),
            "metadata": {
                "text":        c["text"][:500],
                "paper_title": title,
                "source":      "private",
                "company_id":  company_id,
                "filename":    filename,
            }
        } for c, e in zip(chunks, embeddings)]

        for i in range(0, len(vectors), 100):
            idx.upsert(vectors=vectors[i:i+100], namespace=namespace)

        # Update chunk count in Supabase
        _sb().table("private_companies").update({
            "chunk_count": (company.get("chunk_count") or 0) + len(chunks)
        }).eq("company_id", company_id).execute()

        # Save chunk records to Supabase
        now = datetime.now().isoformat()
        chunk_rows = [{
            "chunk_id":   c["chunk_id"],
            "company_id": company_id,
            "text":       c["text"][:1000],
            "paper_title": title,
            "source":     "private",
            "filename":   filename,
            "created_at": now,
        } for c in chunks]

        for i in range(0, len(chunk_rows), 50):
            try:
                _sb().table("private_chunks").upsert(chunk_rows[i:i+50]).execute()
            except: pass

        return {
            "status":       "success",
            "chunks_added": len(chunks),
            "namespace":    namespace,
            "title":        title,
            "filename":     filename,
        }
    except Exception as e:
        return {"error": str(e)}


def search_private(company_id: str, query: str, top_k: int = 10) -> List[Dict]:
    company = get_company(company_id)
    if not company:
        return []
    try:
        import sys; sys.path.insert(0,'.')
        from dotenv import load_dotenv; load_dotenv('.env')
        from learning_module.embedding_bridge import EmbeddingBridge
        from pinecone import Pinecone

        encoder = EmbeddingBridge()
        pc  = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        idx = pc.Index(os.environ['PINECONE_INDEX'])

        qemb    = encoder.embed([query])[0].tolist()
        results = idx.query(
            vector=qemb, top_k=top_k,
            namespace=company["namespace"],
            include_metadata=True
        )
        return [{
            "text":        m.metadata.get("text",""),
            "paper_title": m.metadata.get("paper_title",""),
            "source":      "private",
            "score":       m.score,
            "filename":    m.metadata.get("filename",""),
        } for m in results.matches]
    except Exception as e:
        print(f"search_private error: {e}")
        return []


def get_company_stats(company_id: str) -> Dict:
    company = get_company(company_id)
    if not company:
        return {}
    try:
        docs_r = _sb().table("private_chunks").select(
            "filename, paper_title"
        ).eq("company_id", company_id).execute()

        # Group by filename
        seen, docs = {}, []
        for row in (docs_r.data or []):
            fn = row.get("filename","")
            if fn not in seen:
                seen[fn] = {"filename": fn, "title": row.get("paper_title",""), "chunks": 0}
            seen[fn]["chunks"] += 1
        docs = list(seen.values())
    except:
        docs = []

    return {
        "company_name": company.get("company_name",""),
        "namespace":    company.get("namespace",""),
        "total_chunks": company.get("chunk_count", 0),
        "domain":       company.get("domain","ml"),
        "created_at":   company.get("created_at",""),
        "documents":    docs,
    }
