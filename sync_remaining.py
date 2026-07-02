
import sqlite3, os, sys, time
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env")
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

encoder = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
idx = pc.Index(os.environ.get("PINECONE_INDEX", "tattva"))

db = sqlite3.connect("knowledge_base/knowledge.db")
chunks = db.execute("SELECT chunk_id, text, paper_title, domain FROM chunks WHERE chunk_id > 318514 AND LENGTH(text) > 50 ORDER BY chunk_id").fetchall()
db.close()
print(f"Uploading {len(chunks)} chunks...", flush=True)

BATCH = 100
uploaded = 0
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i+BATCH]
    texts = [c[1][:500] for c in batch]
    vecs = encoder.encode(texts, batch_size=64, show_progress_bar=False)
    upserts = [(f"chunk-{c[0]}", v.tolist(), {"paper_title":(c[2] or "")[:100], "domain":c[3] or "", "text":c[1][:300]}) for c,v in zip(batch,vecs)]
    idx.upsert(vectors=upserts)
    uploaded += len(batch)
    if uploaded % 2000 == 0:
        print(f"Progress: {uploaded}/{len(chunks)}", flush=True)
    time.sleep(0.3)

print(f"DONE. Pinecone: {idx.describe_index_stats().total_vector_count}", flush=True)
