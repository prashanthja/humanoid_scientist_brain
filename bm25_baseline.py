from rank_bm25 import BM25Okapi
import sqlite3, csv, os

# Find the database
db_paths = [
    'knowledge_base/chunks.db',
    'chunks.db',
    'knowledge_base/chunk_store.db',
    'data/chunks.db'
]

db_path = None
for path in db_paths:
    if os.path.exists(path):
        db_path = path
        print(f"Found DB at: {path}")
        break

if not db_path:
    print("DB not found. Listing knowledge_base folder:")
    os.system("ls -la knowledge_base/")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Try different table/column names
try:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")
    
    cursor.execute(f"PRAGMA table_info({tables[0][0]})")
    cols = cursor.fetchall()
    print(f"Columns: {[c[1] for c in cols]}")
    
    table = tables[0][0]
    text_col = next(c[1] for c in cols if c[1] in ['text','content','chunk_text','body'])
    cursor.execute(f"SELECT {text_col} FROM {table} LIMIT 5")
    print(f"Sample rows: {cursor.fetchall()[:2]}")
    cursor.execute(f"SELECT {text_col} FROM {table}")
    rows = cursor.fetchall()
except Exception as e:
    print(f"Error: {e}")
    exit()

conn.close()

chunks = [r[0] for r in rows if r[0]]
print(f"\nLoaded {len(chunks)} chunks")

tokenized = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)

queries = [
    "Does LoRA reduce training cost compared to full fine-tuning?",
    "Does FlashAttention reduce GPU memory usage during training?",
    "Does speculative decoding improve inference throughput?",
    "Does quantization to INT8 degrade model accuracy?",
    "Does Mixture of Experts reduce compute cost per token?",
    "Does KV cache compression reduce memory overhead in long context?",
    "Does grouped query attention reduce inference latency?",
    "Does prefix caching improve throughput for repeated prompts?",
    "Does sliding window attention scale to longer sequences?",
    "Does PagedAttention reduce memory fragmentation in LLM serving?",
    "Mamba vs Transformer: which has lower inference latency at long context?",
    "LoRA vs full fine-tuning: which achieves better task accuracy?",
    "FlashAttention vs standard attention: memory usage comparison",
    "Speculative decoding vs beam search: latency tradeoff comparison",
    "MoE vs dense transformer: throughput per parameter comparison",
    "INT4 quantization vs INT8 quantization: accuracy vs speed tradeoff",
    "RWKV vs Transformer: which scales better with context length?",
    "PagedAttention vs standard KV cache: memory efficiency comparison",
    "GQA vs MHA: latency comparison at inference time",
    "LoRA rank 4 vs rank 16: which gives better performance on downstream tasks?",
    "How does FlashAttention reduce memory usage in transformer attention?",
    "How does speculative decoding achieve speedup without quality loss?",
    "Why does KV cache compression cause accuracy degradation?",
    "How does LoRA achieve parameter efficiency in fine-tuning?",
    "Why does Mamba perform well on long sequences compared to Transformers?",
    "How does mixture of experts routing affect training stability?",
    "Why does quantization-aware training preserve accuracy better than PTQ?",
    "How does continuous batching improve GPU utilization in LLM serving?",
    "Why does sliding window attention fail on tasks requiring global context?",
    "How does tensor parallelism affect communication overhead in multi-GPU inference?"
]

results = []
for q in queries:
    tokens = q.lower().split()
    scores = bm25.get_scores(tokens)
    top10_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    avg_score = sum(scores[i] for i in top10_idx) / 10
    results.append({'query': q[:60], 'avg_bm25_score': round(avg_score, 3)})
    print(f"Q: {q[:55]}... score: {avg_score:.3f}")

with open('bm25_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['query','avg_bm25_score'])
    writer.writeheader()
    writer.writerows(results)

mean_score = sum(r['avg_bm25_score'] for r in results) / len(results)
print(f"\nMean BM25 score across 30 queries: {mean_score:.3f}")
print("Saved to bm25_results.csv")
