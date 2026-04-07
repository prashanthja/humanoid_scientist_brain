#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowledge_base.chunk_store import ChunkStore
from learning_module.trainer_online import OnlineTrainer
from learning_module.embedding_bridge import EmbeddingBridge
from retrieval.chunk_index import ChunkIndex

def rebuild_index():
    print("Rebuilding ChunkIndex...")
    cs = ChunkStore()
    print(f"Chunks in store: {cs.count()}")
    encoder = EmbeddingBridge(OnlineTrainer())
    index = ChunkIndex(encoder=encoder, chunk_store=cs)
    index.rebuild()
    print("Index rebuilt successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild_only", action="store_true")
    parser.add_argument("--rebuild_index", action="store_true")
    args = parser.parse_args()
    rebuild_index()
