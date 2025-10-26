# learning_module/trainer_online.py
# ------------------------------------------------------------
# Continual Transformer (PyTorch) with Adaptive Byte-Pair Tokenizer
# - InfoNCE self-supervised continual training
# - Per-epoch metrics logged to logs/training_history.jsonl
# - Automatic MPS CPU fallback for unsupported ops
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # visual progress bar


# =========================
# Device Utilities
# =========================

def get_device() -> torch.device:
    """Return best available device with MPS→CPU fallback handling."""
    try:
        if torch.cuda.is_available():
            print("⚙️ Using CUDA GPU")
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            # Validate by running a tiny tensor
            try:
                _ = torch.zeros(1).to("mps")
                print("⚙️ Using Apple MPS (Metal) with CPU fallback enabled")
                return torch.device("mps")
            except Exception as e:
                print(f"⚠️ MPS initialization failed ({e}); using CPU instead.")
                return torch.device("cpu")

        print("⚙️ Using CPU backend")
        return torch.device("cpu")

    except Exception as e:
        print(f"⚠️ Device detection failed: {e}")
        return torch.device("cpu")


def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Tokenizer
# =========================

SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3, "MASK": 4}
SPECIAL_INV = {v: k for k, v in SPECIAL_TOKENS.items()}
SPECIAL_COUNT = len(SPECIAL_TOKENS)
TOKENIZER_DIR = "models"
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer_bpe.json")


@dataclass
class BPETokenizerConfig:
    target_vocab: int = 4096
    max_merges_per_fit: int = 400
    min_pair_freq: int = 4


class AdaptiveBPETokenizer:
    """Byte-Pair tokenizer with incremental merge learning."""
    def __init__(self, config: BPETokenizerConfig = BPETokenizerConfig()):
        self.cfg = config
        self.byte_offset = SPECIAL_COUNT
        self.vocab_size = SPECIAL_COUNT + 256
        self.merges: Dict[str, int] = {}
        os.makedirs(TOKENIZER_DIR, exist_ok=True)
        self._load_if_exists()

    def _load_if_exists(self):
        if not os.path.exists(TOKENIZER_PATH):
            return
        try:
            with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab_size = data["vocab_size"]
            self.merges = {k: int(v) for k, v in data["merges"].items()}
        except Exception:
            self.vocab_size = SPECIAL_COUNT + 256
            self.merges = {}

    def _save(self):
        with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges}, f, indent=2)

    def _bytes_to_ids(self, text: str) -> List[int]:
        return [b + self.byte_offset for b in text.encode("utf-8", errors="replace")]

    def _pair_key(self, a: int, b: int) -> str:
        return f"{a},{b}"

    def fit(self, texts: Sequence[str]):
        """Incrementally update merge rules."""
        if self.vocab_size >= self.cfg.target_vocab:
            return
        pair_freq: Dict[Tuple[int, int], int] = {}
        for t in texts:
            if not t:
                continue
            ids = self._bytes_to_ids(t)
            ids = self._apply_merges(ids)
            for a, b in zip(ids, ids[1:]):
                pair_freq[(a, b)] = pair_freq.get((a, b), 0) + 1

        merges_added = 0
        while (self.vocab_size < self.cfg.target_vocab) and (merges_added < self.cfg.max_merges_per_fit):
            if not pair_freq:
                break
            (a, b), freq = max(pair_freq.items(), key=lambda kv: kv[1])
            if freq < self.cfg.min_pair_freq:
                break
            key = self._pair_key(a, b)
            if key in self.merges:
                pair_freq.pop((a, b), None)
                continue
            self.merges[key] = self.vocab_size
            self.vocab_size += 1
            merges_added += 1
            pair_freq.pop((a, b), None)

        if merges_added > 0:
            self._save()

    def encode(self, text: str, add_special=True, max_len: int = 256) -> List[int]:
        ids = self._bytes_to_ids(text or "")
        out = [SPECIAL_TOKENS["BOS"]] + self._apply_merges(ids) + [SPECIAL_TOKENS["EOS"]] if add_special else self._apply_merges(ids)
        if len(out) > max_len:
            out = out[: max_len - 1] + [SPECIAL_TOKENS["EOS"]]
        return out

    def _apply_merges(self, ids: List[int]) -> List[int]:
        if not ids or not self.merges:
            return ids
        merged = True
        while merged:
            merged = False
            i = 0
            out = []
            while i < len(ids):
                if i < len(ids) - 1:
                    key = self._pair_key(ids[i], ids[i + 1])
                    new_id = self.merges.get(key)
                    if new_id is not None:
                        out.append(new_id)
                        i += 2
                        merged = True
                        continue
                out.append(ids[i])
                i += 1
            ids = out
        return ids


# =========================
# Transformer Encoder
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ContinualTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_heads=4, n_layers=4, ff_mult=4, dropout=0.1, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        h = self.tok_emb(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=~attn_mask if attn_mask is not None else None)
        return h

    def sentence_embedding(self, x, attn_mask):
        h = self.forward(x, attn_mask)
        bos = h[:, 0, :]
        z = F.normalize(self.proj(bos), dim=-1)
        return z


# =========================
# Helper Functions
# =========================

def make_views(tokens, mask, drop_prob=0.06, span_mask_prob=0.06):
    B, L = tokens.size()
    out = tokens.clone()
    if drop_prob > 0:
        keep = torch.rand(B, L, device=out.device) > drop_prob
        keep[:, 0] = True
        eos_id = SPECIAL_TOKENS["EOS"]
        eos_pos = (out == eos_id).int().argmax(dim=1)
        for b in range(B):
            keep[b, eos_pos[b].item()] = True
        out = torch.where(keep, out, torch.tensor(SPECIAL_TOKENS["PAD"], device=out.device))
    if span_mask_prob > 0:
        mask_id = SPECIAL_TOKENS["MASK"]
        rnd = torch.rand(B, L, device=out.device)
        span_mask = (rnd < span_mask_prob) & mask
        out = torch.where(span_mask, torch.tensor(mask_id, device=out.device), out)
    return out


def collate_batch(batch_texts, tokenizer, max_len, device):
    ids = [tokenizer.encode(t, add_special=True, max_len=max_len) for t in batch_texts]
    L = min(max(len(x) for x in ids), max_len)
    pad_id = SPECIAL_TOKENS["PAD"]
    toks = torch.full((len(ids), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.bool)
    for i, seq in enumerate(ids):
        s = seq[:L]
        toks[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = True
    return toks.to(device), attn.to(device)


def avg_in_batch_cosine(z1, z2):
    with torch.no_grad():
        return float((z1 * z2).sum(dim=-1).mean().item())


# =========================
# Online Trainer
# =========================

class OnlineTrainer:
    def __init__(self, encoder=None, batch_size=32, epochs=2, lr=3e-4, max_len=256,
                 model_dim=256, n_heads=4, n_layers=4, target_vocab=4096):
        set_seed(1337)
        self.device = get_device()
        self.batch_size, self.epochs, self.lr, self.max_len = batch_size, epochs, lr, max_len

        self.tokenizer = AdaptiveBPETokenizer(BPETokenizerConfig(target_vocab=target_vocab))
        self.model = ContinualTransformerEncoder(self.tokenizer.vocab_size, model_dim, n_heads, n_layers, max_len=max_len).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.last_avg_loss = None
        self.last_similarity = None
        self.epoch_history: list[dict] = []
        os.makedirs("logs", exist_ok=True)
        self.metrics_log_path = "logs/training_history.jsonl"

        os.makedirs("models", exist_ok=True)
        self.ckpt_path = "models/continual_transformer.pt"
        self._try_load_ckpt()

    # ---------- persistence ----------
    def _try_load_ckpt(self):
        if not os.path.exists(self.ckpt_path):
            return
        try:
            payload = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(payload["model"], strict=False)
            self.opt.load_state_dict(payload["opt"])
            print("💾 Loaded continual transformer checkpoint.")
        except Exception:
            pass

    def _save_ckpt(self):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "vocab_size": self.tokenizer.vocab_size}, self.ckpt_path)

    def _maybe_resize_embeddings(self):
        if self.model.vocab_size == self.tokenizer.vocab_size:
            return
        old = self.model.tok_emb
        new = nn.Embedding(self.tokenizer.vocab_size, old.embedding_dim).to(self.device)
        with torch.no_grad():
            n = min(old.num_embeddings, new.num_embeddings)
            new.weight[:n].copy_(old.weight[:n])
        self.model.tok_emb = new
        self.model.vocab_size = self.tokenizer.vocab_size

    # ---------- main ----------
    def incremental_train(self, new_items: Sequence[str | Dict[str, Any]]):
        texts = [i if isinstance(i, str) else i.get("text", "") for i in new_items]
        texts = [t.strip() for t in texts if t.strip()]
        if len(texts) < 4:
            return

        self.tokenizer.fit(texts)
        self._maybe_resize_embeddings()
        self.model.train()
        self.epoch_history = []

        print(f"\n🚀 Training on {len(texts)} new samples for {self.epochs} epochs...")
        for ep in range(1, self.epochs + 1):
            random.shuffle(texts)
            ep_loss, ep_sim, ep_acc, steps = 0, 0, 0, 0
            start_time = time.time()
            with tqdm(total=len(texts), desc=f"Epoch {ep}/{self.epochs}", ncols=80) as pbar:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    if len(batch) < 2:
                        continue
                    toks, attn = collate_batch(batch, self.tokenizer, self.max_len, self.device)
                    t1, t2 = make_views(toks, attn), make_views(toks, attn)
                    z1, z2 = self.model.sentence_embedding(t1, attn), self.model.sentence_embedding(t2, attn)

                    logits = (z1 @ z2.t()) / 0.07
                    labels = torch.arange(logits.size(0), device=self.device)
                    loss = F.cross_entropy(logits, labels)

                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()

                    pred = logits.argmax(dim=1)
                    acc = (pred == labels).float().mean().item()
                    ep_loss += float(loss.item())
                    ep_sim += avg_in_batch_cosine(z1, z2)
                    ep_acc += acc
                    steps += 1
                    pbar.update(len(batch))
                    pbar.set_postfix(loss=f"{loss.item():.3f}")

            duration = time.time() - start_time
            if steps > 0:
                rec = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": ep,
                    "loss": ep_loss / steps,
                    "similarity": ep_sim / steps,
                    "accuracy": ep_acc / steps,
                    "vocab_size": self.tokenizer.vocab_size,
                    "device": self.device.type,
                    "duration_s": round(duration, 2),
                }
                self.epoch_history.append(rec)
                with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                print(f"✅ Epoch {ep}/{self.epochs} finished — "
                      f"loss={rec['loss']:.4f} sim={rec['similarity']:.4f} "
                      f"acc={rec['accuracy']:.4f} ({rec['duration_s']}s)")

        if self.epoch_history:
            self.last_avg_loss = self.epoch_history[-1]["loss"]
            self.last_similarity = self.epoch_history[-1]["similarity"]

        self._save_ckpt()
        print(f"🧪 Online train: epochs={self.epochs}, last_loss={self.last_avg_loss:.4f} "
              f"sim={self.last_similarity:.4f}, vocab={self.tokenizer.vocab_size}, device={self.device.type}")

    # ---------- embedding interface ----------
    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> "np.ndarray":
        import numpy as np
        self.model.eval()
        clean = [t if isinstance(t, str) else str(t) for t in texts if t]
        if not clean:
            return np.zeros((0, self.model.d_model), dtype=np.float32)

        toks, attn = collate_batch(clean, self.tokenizer, self.max_len, self.device)
        z = self.model.sentence_embedding(toks, attn)
        return z.detach().cpu().numpy().astype("float32")
