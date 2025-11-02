# learning_module/trainer_online.py
# ------------------------------------------------------------
# Universal Continual Transformer (PyTorch) with Adaptive BPE
# - Dual Objective: InfoNCE (contrastive) + MLM (masked LM)
# - Continual learning with vocab growth & safe resize
# - Per-epoch metrics → logs/training_history.jsonl
# - MPS CPU fallback; AMP on CUDA
# - Adaptive Epoch Scaling (self-tuning)
# - Env-tunable model size (see notes near top)
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
from tqdm import tqdm


# =========================
# Runtime / Device Utilities
# =========================

def get_device() -> torch.device:
    """Return best available device with MPS→CPU fallback handling."""
    try:
        if torch.cuda.is_available():
            print("⚙️ Using CUDA GPU")
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
    target_vocab: int = int(os.environ.get("UT_TARGET_VOCAB", "4096"))
    max_merges_per_fit: int = 400
    min_pair_freq: int = 4


class AdaptiveBPETokenizer:
    """Byte-Pair tokenizer with incremental merge learning (byte-level base)."""

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
            self.vocab_size = int(data["vocab_size"])
            self.merges = {k: int(v) for k, v in data["merges"].items()}
        except Exception:
            self.vocab_size = SPECIAL_COUNT + 256
            self.merges = {}

    def _save(self):
        with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges}, f, indent=2)

    def _bytes_to_ids(self, text: str) -> List[int]:
        return [b + self.byte_offset for b in (text or "").encode("utf-8", errors="replace")]

    def _pair_key(self, a: int, b: int) -> str:
        return f"{a},{b}"

    def fit(self, texts: Sequence[str]):
        """Incrementally learn merges up to target vocab."""
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

    def encode(self, text: str, add_special=True, max_len: int = 512) -> List[int]:
        ids = self._bytes_to_ids(text or "")
        merged = self._apply_merges(ids)
        if add_special:
            out = [SPECIAL_TOKENS["BOS"]] + merged + [SPECIAL_TOKENS["EOS"]]
        else:
            out = merged
        if len(out) > max_len:
            out = out[: max_len - 1] + [SPECIAL_TOKENS["EOS"]]
        return out

    def _apply_merges(self, ids: List[int]) -> List[int]:
        if not ids or not self.merges:
            return ids
        merged = True
        while merged:
            merged = False
            i, out = 0, []
            while i < len(ids):
                if i < len(ids) - 1:
                    key = self._pair_key(ids[i], ids[i + 1])
                    new_id = self.merges.get(key)
                    if new_id is not None:
                        out.append(new_id)
                        i += 2
                        merged = True
                        continue
                out.append(ids[i]); i += 1
            ids = out
        return ids


# =========================
# Model
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ContinualTransformerEncoder(nn.Module):
    """
    Deep encoder with:
      - Token embedding + sinusoidal PE
      - TransformerEncoder stack
      - Projection head for embeddings
      - LM head (tied weights) for MLM objective
    """

    def __init__(
        self,
        vocab_size: int,
        d_model=512,
        n_heads=8,
        n_layers=8,
        ff_mult=4,
        dropout=0.1,
        max_len=2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model * ff_mult,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

        # LM head for MLM (weight-tying with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self):
        # tie lm_head weight to token embedding
        self.lm_head.weight = self.tok_emb.weight

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

    def mlm_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(h)


# =========================
# Helper Functions
# =========================

def _mask_for_mlm(tokens: torch.Tensor, attn: torch.Tensor, mlm_prob=0.12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM inputs and targets:
      - 12% of visible (non-special) positions are masked for prediction
      - 80% MASK, 10% random token, 10% keep (BERT-style light heuristic)
    """
    device = tokens.device
    B, L = tokens.size()
    maskable = attn.clone()
    # Do not mask BOS/EOS
    maskable[:, 0] = False
    eos_id = SPECIAL_TOKENS["EOS"]
    eos_positions = (tokens == eos_id).int().argmax(dim=1)
    for b in range(B):
        maskable[b, eos_positions[b]] = False

    prob = torch.rand(B, L, device=device)
    mlm_positions = (prob < mlm_prob) & maskable
    targets = tokens.clone()
    targets[~mlm_positions] = -100  # ignore index

    # Apply 80/10/10
    mask_id = SPECIAL_TOKENS["MASK"]
    rand = torch.rand(B, L, device=device)
    # 80% --> MASK
    mask80 = (rand < 0.8) & mlm_positions
    # 10% --> random token (excluding specials)
    rand10 = (rand >= 0.8) & (rand < 0.9) & mlm_positions
    # 10% --> keep original (do nothing)

    tokens_masked = tokens.clone()
    tokens_masked[mask80] = mask_id
    if rand10.any():
        # random IDs in [SPECIAL_COUNT, vocab_size)
        low = SPECIAL_COUNT
        high = int(tokens.max().item() + 1)  # quick bound; safe but not perfect
        if high <= low: high = low + 1
        tokens_masked[rand10] = torch.randint(low, high, size=(rand10.sum(),), device=device)

    return tokens_masked, targets


def make_views(tokens, mask, drop_prob=0.06, span_mask_prob=0.06):
    """Augmentations for contrastive views (keeps BOS/EOS)."""
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
        toks[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = True
    return toks.to(device), attn.to(device)


def avg_in_batch_cosine(z1, z2):
    with torch.no_grad():
        return float((z1 * z2).sum(dim=-1).mean().item())


# =========================
# Online Trainer
# =========================

class OnlineTrainer:
    """
    Universal continual trainer with dual loss:
      - InfoNCE contrastive loss across two augmented views
      - MLM token prediction loss on masked positions
    Exposes:
      - incremental_train(new_items)
      - embed(texts) -> np.ndarray [N, D]
    """

    def __init__(
        self,
        encoder=None,
        batch_size: int = int(os.environ.get("UT_BATCH", "24")),
        epochs: int = int(os.environ.get("UT_EPOCHS", "2")),
        lr: float = float(os.environ.get("UT_LR", "3e-4")),
        max_len: int = int(os.environ.get("UT_MAXLEN", "512")),
        model_dim: int = int(os.environ.get("UT_MODEL_DIM", "512")),
        n_heads: int = int(os.environ.get("UT_N_HEADS", "8")),
        n_layers: int = int(os.environ.get("UT_N_LAYERS", "8")),
        target_vocab: int = int(os.environ.get("UT_TARGET_VOCAB", "4096")),
        use_checkpointing: bool = bool(int(os.environ.get("UT_CHECKPOINT", "0"))),
        alpha_infonce: float = float(os.environ.get("UT_ALPHA_NCE", "1.0")),
        beta_mlm: float = float(os.environ.get("UT_BETA_MLM", "0.7")),
    ):
        set_seed(1337)
        self.device = get_device()
        self.batch_size, self.epochs, self.lr, self.max_len = batch_size, epochs, lr, max_len

        self.alpha_infonce = alpha_infonce
        self.beta_mlm = beta_mlm

        self.tokenizer = AdaptiveBPETokenizer(BPETokenizerConfig(target_vocab=target_vocab))
        self.model = ContinualTransformerEncoder(
            self.tokenizer.vocab_size,
            d_model=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
        ).to(self.device)

        # Optional gradient checkpointing
        if use_checkpointing:
            try:
                for m in self.model.enc.layers:
                    m.gradient_checkpointing = True  # harmless flag; PyTorch uses torch.utils.checkpoint if you wrap
            except Exception:
                pass  # keep running; we aren't wrapping submodules here to avoid extra deps

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.last_avg_loss = None
        self.last_similarity = None
        self.prev_loss = None
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
            # resize if tokenizer grew
            if payload.get("vocab_size") and payload["vocab_size"] != self.tokenizer.vocab_size:
                self._maybe_resize_embeddings()
            print("💾 Loaded continual transformer checkpoint.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    def _save_ckpt(self):
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "vocab_size": self.tokenizer.vocab_size
        }, self.ckpt_path)

    def _maybe_resize_embeddings(self):
        """Resize token embedding & LM head if vocab grew."""
        if self.model.vocab_size == self.tokenizer.vocab_size:
            return
        old_tok = self.model.tok_emb
        new_tok = nn.Embedding(self.tokenizer.vocab_size, old_tok.embedding_dim).to(self.device)
        with torch.no_grad():
            n = min(old_tok.num_embeddings, new_tok.num_embeddings)
            new_tok.weight[:n].copy_(old_tok.weight[:n])
            if n < new_tok.num_embeddings:
                nn.init.normal_(new_tok.weight[n:], mean=0.0, std=0.02)
        self.model.tok_emb = new_tok
        self.model.vocab_size = self.tokenizer.vocab_size

        # lm_head is tied — re-tie after replacing tok_emb
        self.model._tie_weights()

    # ---------- adaptive epoch scaling ----------
    def adjust_epochs(self):
        """Adaptively adjust epoch count based on loss improvements."""
        if self.last_avg_loss is None:
            return self.epochs
        if self.prev_loss is None:
            self.prev_loss = self.last_avg_loss
            return self.epochs

        prev, curr = self.prev_loss, self.last_avg_loss
        improvement = (prev - curr) / max(prev, 1e-6)
        if improvement < 0.02:
            self.epochs = min(self.epochs + 1, 8)
            print(f"🧩 Slow improvement ({improvement:.2%}) → increasing epochs → {self.epochs}")
        elif improvement > 0.15:
            self.epochs = max(self.epochs - 1, 2)
            print(f"⚡ Strong improvement ({improvement:.2%}) → reducing epochs → {self.epochs}")
        else:
            print(f"🔁 Stable training ({improvement:.2%}) → keeping epochs = {self.epochs}")
        self.prev_loss = curr
        return self.epochs

    # ---------- main ----------
    def incremental_train(self, new_items: Sequence[str | Dict[str, Any]]):
        # Normalize inputs
        texts = [i if isinstance(i, str) else (i.get("text") or "") for i in new_items]
        texts = [t.strip() for t in texts if isinstance(t, str) and t and t.strip()]
        if len(texts) < 4:
            return

        # Tokenizer growth and model resize if needed
        self.tokenizer.fit(texts)
        self._maybe_resize_embeddings()

        self.model.train()
        self.epoch_history = []

        epochs_to_run = self.epochs
        print(f"\n🚀 Training on {len(texts)} new samples for {epochs_to_run} epochs...")
        for ep in range(1, epochs_to_run + 1):
            random.shuffle(texts)
            ep_loss, ep_sim, ep_acc, steps = 0.0, 0.0, 0.0, 0
            t0 = time.time()

            with tqdm(total=len(texts), desc=f"Epoch {ep}/{epochs_to_run}", ncols=80) as pbar:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i + self.batch_size]
                    if len(batch) < 2:
                        continue

                    toks, attn = collate_batch(batch, self.tokenizer, self.max_len, self.device)

                    # -------- Contrastive views --------
                    t1 = make_views(toks, attn)
                    t2 = make_views(toks, attn)

                    # -------- MLM prep --------
                    mlm_inputs, mlm_targets = _mask_for_mlm(toks, attn, mlm_prob=0.12)

                    # AMP only on CUDA (MPS/CPU stays FP32)
                    use_amp = (self.device.type == "cuda")
                    ctx = torch.cuda.amp.autocast(enabled=use_amp)

                    with ctx:
                        # Contrastive embeddings
                        z1 = self.model.sentence_embedding(t1, attn)
                        z2 = self.model.sentence_embedding(t2, attn)
                        logits = (z1 @ z2.t()) / 0.07
                        labels = torch.arange(logits.size(0), device=self.device)
                        loss_nce = F.cross_entropy(logits, labels)

                        # MLM logits on full sequence
                        h = self.model(mlm_inputs, attn)
                        vocab_logits = self.model.mlm_logits(h)
                        loss_mlm = F.cross_entropy(
                            vocab_logits.view(-1, self.tokenizer.vocab_size),
                            mlm_targets.view(-1),
                            ignore_index=-100
                        )

                        # Combined loss
                        loss = self.alpha_infonce * loss_nce + self.beta_mlm * loss_mlm

                    self.opt.zero_grad(set_to_none=True)
                    if use_amp:
                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()

                    # simple alignment accuracy for logging
                    pred = logits.argmax(dim=1)
                    acc = (pred == labels).float().mean().item()

                    ep_loss += float(loss.item())
                    ep_sim += avg_in_batch_cosine(z1, z2)
                    ep_acc += acc
                    steps += 1

                    pbar.update(len(batch))
                    pbar.set_postfix(loss=f"{loss.item():.3f}", nce=f"{loss_nce.item():.3f}", mlm=f"{loss_mlm.item():.3f}")

            duration = time.time() - t0
            if steps > 0:
                rec = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": ep,
                    "loss_total": ep_loss / steps,
                    "similarity": ep_sim / steps,
                    "align_acc": ep_acc / steps,
                    "alpha_infonce": self.alpha_infonce,
                    "beta_mlm": self.beta_mlm,
                    "vocab_size": self.tokenizer.vocab_size,
                    "device": self.device.type,
                    "duration_s": round(duration, 2),
                }
                self.epoch_history.append(rec)
                with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                print(f"✅ Epoch {ep}/{epochs_to_run} — "
                      f"loss={rec['loss_total']:.4f} sim={rec['similarity']:.4f} "
                      f"acc={rec['align_acc']:.4f} ({rec['duration_s']}s)")

        if self.epoch_history:
            self.last_avg_loss = self.epoch_history[-1]["loss_total"]
            self.last_similarity = self.epoch_history[-1]["similarity"]

        self._save_ckpt()
        print(f"🧪 Online train: epochs={epochs_to_run}, last_loss={self.last_avg_loss:.4f} "
              f"sim={self.last_similarity:.4f}, vocab={self.tokenizer.vocab_size}, device={self.device.type}")

        # Adjust epochs for next call (self-tuning)
        self.adjust_epochs()

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
