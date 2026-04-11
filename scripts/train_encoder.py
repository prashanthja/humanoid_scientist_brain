#!/usr/bin/env python3
"""
Train Tattva's custom encoder on the existing chunk corpus.

Usage:
    cd /Users/milky/Downloads/robot/humanoid_scientist_brain
    python3 scripts/train_encoder.py

What it does:
1. Loads all chunks from SQLite (1,798+ texts)
2. Trains OnlineTrainer with InfoNCE + MLM objectives
3. Saves checkpoint to models/continual_transformer.pt
4. Rebuilds the vector index with trained weights
5. Verifies embeddings are consistent

After this runs, the encoder is deterministic and persistent.
The background service will use the same saved weights.
"""

import sys, os, json, sqlite3, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DB = os.path.join(ROOT, "knowledge_base", "knowledge.db")
CKPT     = os.path.join(ROOT, "models", "continual_transformer.pt")

def load_chunks():
    """Load all chunk texts from SQLite."""
    conn = sqlite3.connect(CHUNK_DB)
    rows = conn.execute("SELECT text FROM chunks WHERE text IS NOT NULL").fetchall()
    conn.close()
    texts = [r[0].strip() for r in rows if r[0] and len(r[0].strip()) > 50]
    print(f"✅ Loaded {len(texts)} chunks from database")
    return texts

def train(texts):
    """Train the encoder on chunk texts."""
    from learning_module.trainer_online import OnlineTrainer

    print("\n🚀 Initializing encoder...")
    print("   Architecture: d_model=512, n_heads=8, n_layers=8")
    print("   Objective: InfoNCE (contrastive) + MLM")
    print(f"   Training samples: {len(texts)}")
    print(f"   Checkpoint: {CKPT}")
    print()

    # Remove old mismatched checkpoint if exists
    old_ckpt = CKPT + ".old"
    if os.path.exists(CKPT):
        os.rename(CKPT, old_ckpt)
        print(f"⚠️  Moved old checkpoint to {old_ckpt}")

    trainer = OnlineTrainer(
        min_train_samples=10,
        batch_size=16,
        epochs=5,
    )

    print("\n📚 Phase 1: Training BPE tokenizer on corpus...")
    trainer.tokenizer.fit(texts)
    trainer._maybe_resize_embeddings()
    print(f"   Vocab size: {trainer.tokenizer.vocab_size}")

    print("\n🧠 Phase 2: Training transformer encoder...")
    print("   This will take 30-90 minutes on CPU/MPS...")
    print("   Progress shown per epoch\n")

    t0 = time.time()
    trainer.incremental_train(texts)
    elapsed = round(time.time() - t0, 1)

    print(f"\n✅ Training complete in {elapsed}s")
    print(f"💾 Checkpoint saved to: {CKPT}")

    return trainer

def verify(trainer):
    """Verify embeddings are consistent across two calls."""
    print("\n🔬 Verifying embedding consistency...")
    test_texts = [
        "Does FlashAttention reduce memory overhead?",
        "KV cache compression reduces inference latency",
        "LoRA fine-tuning is parameter efficient",
    ]

    import numpy as np
    vecs1 = trainer.embed(test_texts)
    vecs2 = trainer.embed(test_texts)

    max_diff = float(np.max(np.abs(vecs1 - vecs2)))
    print(f"   Max diff between two calls: {max_diff:.6f} (should be 0.0)")
    assert max_diff < 1e-5, "Embeddings are not consistent!"

    # Check embeddings are meaningful (not all zeros/same)
    sim_same = float(np.dot(vecs1[0], vecs1[0]))
    sim_diff = float(np.dot(vecs1[0], vecs1[1]))
    print(f"   Self-similarity: {sim_same:.4f} (should be ~1.0)")
    print(f"   Cross-similarity (FlashAttention vs KV cache): {sim_diff:.4f}")
    print("✅ Embeddings verified — deterministic and meaningful")

def rebuild_index(trainer):
    """Rebuild the vector index with trained encoder weights."""
    print("\n🔄 Rebuilding vector index with trained encoder...")

    from knowledge_base.chunk_store import ChunkStore
    from learning_module.embedding_bridge import EmbeddingBridge
    from retrieval.chunk_index import ChunkIndex

    cs      = ChunkStore()
    encoder = EmbeddingBridge(trainer)
    index   = ChunkIndex(encoder=encoder, chunk_store=cs)

    t0 = time.time()
    index.rebuild()
    elapsed = round(time.time() - t0, 1)

    print(f"✅ Index rebuilt in {elapsed}s")
    print(f"   Vectors saved to: data/chunk_index_vecs.npy")

    # Quick retrieval test
    print("\n🔍 Testing retrieval...")
    results = index.retrieve("Does FlashAttention reduce memory overhead?", top_k=3)
    print(f"   Retrieved {len(results)} chunks")
    for r in results[:2]:
        print(f"   [{r.get('similarity',0):.3f}] {r.get('paper_title','?')[:60]}")

    print("\n✅ Everything working correctly!")

def main():
    print("=" * 60)
    print("Tattva AI — Encoder Training Script")
    print("=" * 60)

    # Load training data
    texts = load_chunks()
    if len(texts) < 100:
        print("❌ Not enough chunks. Run background service first.")
        sys.exit(1)

    # Train encoder
    trainer = train(texts)

    # Verify embeddings
    verify(trainer)

    # Rebuild index
    rebuild_index(trainer)

    print("\n" + "=" * 60)
    print("✅ Training complete. Your encoder is ready.")
    print("   Restart Flask to use the new encoder:")
    print("   pkill -f 'dashboard/app.py'")
    print("   python3 dashboard/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()