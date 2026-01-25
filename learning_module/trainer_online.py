# learning_module/trainer_online.py
# ------------------------------------------------------------
# Universal Continual Transformer (PyTorch) with Adaptive BPE
# - Dual Objective: InfoNCE (contrastive) + MLM (masked LM)
# - Continual learning with vocab growth & safe resize
# - Per-epoch metrics → logs/training_history.jsonl
# - MPS CPU fallback; AMP on CUDA (torch.amp.* API)
# - Adaptive Epoch Scaling (self-tuning)
# - Checkpoint stores model_config; loader rebuilds model to avoid shape mismatch
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Sequence, Optional

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

    def encode(self, text: str, add_special: bool = True, max_len: int = 512) -> List[int]:
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
                out.append(ids[i])
                i += 1
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
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos = PositionalEncoding(self.d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            self.d_model,
            n_heads,
            self.d_model * ff_mult,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh())

        # LM head for MLM (weight-tying with token embedding)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self):
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.tok_emb(x)
        h = self.pos(h)
        # src_key_padding_mask expects True for padding positions
        h = self.enc(h, src_key_padding_mask=(~attn_mask) if attn_mask is not None else None)
        return h
    
    def sentence_embedding(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            h = self.forward(x, attn_mask)  # [B, L, D]

            # masked mean pooling over real tokens (prevents BOS collapse)
            m = attn_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B, D]

            z = F.normalize(self.proj(pooled), dim=-1)
            return z

    def mlm_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(h)


# =========================
# Helper Functions
# =========================

def _mask_for_mlm(
    tokens: torch.Tensor,
    attn: torch.Tensor,
    mlm_prob: float = 0.12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM inputs and targets:
      - 12% of visible (non-special) positions are masked for prediction
      - 80% MASK, 10% random token, 10% keep (BERT-style heuristic)
    """
    device = tokens.device
    B, L = tokens.size()

    maskable = attn.clone()
    # Do not mask BOS
    maskable[:, 0] = False

    eos_id = SPECIAL_TOKENS["EOS"]
    eos_positions = (tokens == eos_id).int().argmax(dim=1)
    for b in range(B):
        # Do not mask EOS
        maskable[b, eos_positions[b]] = False

    prob = torch.rand(B, L, device=device)
    mlm_positions = (prob < mlm_prob) & maskable

    targets = tokens.clone()
    targets[~mlm_positions] = -100  # ignore index

    # Apply 80/10/10
    mask_id = SPECIAL_TOKENS["MASK"]
    rand = torch.rand(B, L, device=device)

    mask80 = (rand < 0.8) & mlm_positions
    rand10 = (rand >= 0.8) & (rand < 0.9) & mlm_positions

    tokens_masked = tokens.clone()
    tokens_masked[mask80] = mask_id

    if rand10.any():
        # Random IDs in [SPECIAL_COUNT, vocab_size)
        low = SPECIAL_COUNT
        high = int(tokens.max().item() + 1)
        if high <= low:
            high = low + 1
        tokens_masked[rand10] = torch.randint(low, high, size=(int(rand10.sum().item()),), device=device)

    return tokens_masked, targets


def make_views(tokens: torch.Tensor, mask: torch.Tensor, drop_prob: float = 0.06, span_mask_prob: float = 0.06):
    """Augmentations for contrastive views (keeps BOS/EOS)."""
    B, L = tokens.size()
    out = tokens.clone()

    if drop_prob > 0:
        keep = torch.rand(B, L, device=out.device) > drop_prob
        keep[:, 0] = True

        eos_id = SPECIAL_TOKENS["EOS"]
        eos_pos = (out == eos_id).int().argmax(dim=1)
        for b in range(B):
            keep[b, int(eos_pos[b].item())] = True

        out = torch.where(keep, out, torch.tensor(SPECIAL_TOKENS["PAD"], device=out.device))

    if span_mask_prob > 0:
        mask_id = SPECIAL_TOKENS["MASK"]
        rnd = torch.rand(B, L, device=out.device)
        span_mask = (rnd < span_mask_prob) & mask
        out = torch.where(span_mask, torch.tensor(mask_id, device=out.device), out)

    return out


def collate_batch(batch_texts, tokenizer: AdaptiveBPETokenizer, max_len: int, device: torch.device):
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


def avg_in_batch_cosine(z1: torch.Tensor, z2: torch.Tensor) -> float:
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
        min_train_samples: int = int(os.environ.get("UT_MIN_SAMPLES", "8")),
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

        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.max_len = int(max_len)

        self.alpha_infonce = float(alpha_infonce)
        self.beta_mlm = float(beta_mlm)

        # Keep model hyperparams for ckpt config
        self.model_dim = int(model_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_mult = 4
        self.dropout = 0.1

        self.tokenizer = AdaptiveBPETokenizer(BPETokenizerConfig(target_vocab=int(target_vocab)))

        # Build initial model from current env/config
        self._build_model(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.model_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_len=self.max_len,
        )

        # Optional gradient checkpointing flag (placeholder)
        if use_checkpointing:
            try:
                for m in self.model.enc.layers:
                    m.gradient_checkpointing = True
            except Exception:
                pass

        self.last_avg_loss = None
        self.last_similarity = None
        self.prev_loss = None
        self.epoch_history: list[dict] = []

        os.makedirs("logs", exist_ok=True)
        self.metrics_log_path = "logs/training_history.jsonl"

        os.makedirs("models", exist_ok=True)
        self.ckpt_path = "models/continual_transformer.pt"
        self._try_load_ckpt()

    # ---------- model build / rebuild ----------
    def _build_model(self, *, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int):
        self.model = ContinualTransformerEncoder(
            vocab_size=int(vocab_size),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            ff_mult=int(self.ff_mult),
            dropout=float(self.dropout),
            max_len=int(max_len),
        ).to(self.device)

        # Recreate optimizer for current parameter set
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # AMP scaler (new API)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

    # ---------- persistence ----------
    def _try_load_ckpt(self):
        if not os.path.exists(self.ckpt_path):
            return

        try:
            payload = torch.load(self.ckpt_path, map_location="cpu")
            cfg = payload.get("model_config", None)

            if cfg is None:
                # Backward compatible legacy load (best-effort)
                print("⚠️ Checkpoint missing model_config. Loading best-effort (may reset optimizer).")
                self.model.load_state_dict(payload["model"], strict=False)

                if "opt" in payload:
                    try:
                        self.opt.load_state_dict(payload["opt"])
                    except Exception:
                        print("⚠️ Optimizer state mismatch; skipping optimizer load.")

                if payload.get("vocab_size") and int(payload["vocab_size"]) != int(self.tokenizer.vocab_size):
                    self._maybe_resize_embeddings()

                self.model.to(self.device)
                print("💾 Loaded checkpoint (legacy).")
                return

            # Config-aware restore: rebuild model exactly like ckpt
            ckpt_vocab = int(cfg["vocab_size"])
            ckpt_d = int(cfg["d_model"])
            ckpt_heads = int(cfg["n_heads"])
            ckpt_layers = int(cfg["n_layers"])
            ckpt_max_len = int(cfg.get("max_len", self.max_len))

            self._build_model(
                vocab_size=ckpt_vocab,
                d_model=ckpt_d,
                n_heads=ckpt_heads,
                n_layers=ckpt_layers,
                max_len=ckpt_max_len,
            )

            self.model.load_state_dict(payload["model"], strict=True)

            # Resize embeddings if tokenizer grew beyond ckpt vocab
            if int(self.tokenizer.vocab_size) != int(self.model.vocab_size):
                self._maybe_resize_embeddings()

            # Optimizer is safe to load only if no vocab resize broke param groups
            if "opt" in payload:
                try:
                    self.opt.load_state_dict(payload["opt"])
                except Exception:
                    print("⚠️ Optimizer state mismatch (likely vocab resize); skipping optimizer load.")

            self.model.to(self.device)
            print(f"💾 Loaded continual transformer checkpoint (cfg: d={ckpt_d}, heads={ckpt_heads}, layers={ckpt_layers}, vocab={ckpt_vocab}).")

        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    def _save_ckpt(self):
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "vocab_size": int(self.tokenizer.vocab_size),
            "model_config": {
                "vocab_size": int(self.model.vocab_size),
                "d_model": int(self.model.d_model),
                "n_heads": int(self.n_heads),
                "n_layers": int(self.n_layers),
                "max_len": int(self.max_len),
            },
        }, self.ckpt_path)

    def _maybe_resize_embeddings(self):
        """Resize token embedding & LM head if vocab grew."""
        if int(self.model.vocab_size) == int(self.tokenizer.vocab_size):
            return

        old_tok = self.model.tok_emb
        new_tok = nn.Embedding(int(self.tokenizer.vocab_size), int(old_tok.embedding_dim)).to(self.device)

        with torch.no_grad():
            n = min(int(old_tok.num_embeddings), int(new_tok.num_embeddings))
            new_tok.weight[:n].copy_(old_tok.weight[:n])
            if n < int(new_tok.num_embeddings):
                nn.init.normal_(new_tok.weight[n:], mean=0.0, std=0.02)

        self.model.tok_emb = new_tok
        self.model.vocab_size = int(self.tokenizer.vocab_size)

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

        prev, curr = float(self.prev_loss), float(self.last_avg_loss)
        improvement = (prev - curr) / max(prev, 1e-6)

        if improvement < 0.02:
            self.epochs = min(int(self.epochs) + 1, 8)
            print(f"🧩 Slow improvement ({improvement:.2%}) → increasing epochs → {self.epochs}")
        elif improvement > 0.15:
            self.epochs = max(int(self.epochs) - 1, 2)
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

        epochs_to_run = int(self.epochs)
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

                    use_amp = (self.device.type == "cuda")

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        # Contrastive embeddings
                        z1 = self.model.sentence_embedding(t1, attn)
                        z2 = self.model.sentence_embedding(t2, attn)

                        logits12 = (z1 @ z2.t()) / 0.07
                        logits21 = (z2 @ z1.t()) / 0.07
                        labels = torch.arange(logits12.size(0), device=self.device)

                        loss_nce = 0.5 * (
                            F.cross_entropy(logits12, labels) +
                            F.cross_entropy(logits21, labels)
                        )

                        # MLM logits on full sequence
                        h = self.model(mlm_inputs, attn)
                        vocab_logits = self.model.mlm_logits(h)
                        loss_mlm = F.cross_entropy(
                            vocab_logits.view(-1, int(self.tokenizer.vocab_size)),
                            mlm_targets.view(-1),
                            ignore_index=-100,
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

                    # alignment accuracy (logging)
                    pred = logits12.argmax(dim=1)
                    acc = (pred == labels).float().mean().item()


                    ep_loss += float(loss.item())
                    ep_sim += avg_in_batch_cosine(z1, z2)
                    ep_acc += float(acc)
                    steps += 1

                    pbar.update(len(batch))
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        nce=f"{loss_nce.item():.3f}",
                        mlm=f"{loss_mlm.item():.3f}",
                    )

            duration = time.time() - t0
            if steps > 0:
                rec = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": int(ep),
                    "loss_total": float(ep_loss / steps),
                    "similarity": float(ep_sim / steps),
                    "align_acc": float(ep_acc / steps),
                    "alpha_infonce": float(self.alpha_infonce),
                    "beta_mlm": float(self.beta_mlm),
                    "vocab_size": int(self.tokenizer.vocab_size),
                    "device": str(self.device.type),
                    "duration_s": round(float(duration), 2),
                }
                self.epoch_history.append(rec)

                with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                print(
                    f"✅ Epoch {ep}/{epochs_to_run} — "
                    f"loss={rec['loss_total']:.4f} sim={rec['similarity']:.4f} "
                    f"acc={rec['align_acc']:.4f} ({rec['duration_s']}s)"
                )

        if self.epoch_history:
            self.last_avg_loss = float(self.epoch_history[-1]["loss_total"])
            self.last_similarity = float(self.epoch_history[-1]["similarity"])

        self._save_ckpt()

        print(
            f"🧪 Online train: epochs={epochs_to_run}, last_loss={self.last_avg_loss:.4f} "
            f"sim={self.last_similarity:.4f}, vocab={int(self.tokenizer.vocab_size)}, device={self.device.type}"
        )

        # Adjust epochs for next call (self-tuning)
        self.adjust_epochs()

    # ---------- embedding interface ----------
    @torch.no_grad()
    def embed(self, texts, *, max_len: int = 192, batch_size: int = 16, force_cpu: bool = False):
        import numpy as np

        self.model.eval()
        clean = [t if isinstance(t, str) else str(t) for t in (texts or []) if t]
        if not clean:
            return np.zeros((0, int(self.model.d_model)), dtype=np.float32)

        device = torch.device("cpu") if force_cpu else self.device

        moved = False
        if force_cpu and self.device.type != "cpu":
            self.model.to(device)
            moved = True

        try:
            outs = []
            for i in range(0, len(clean), batch_size):
                batch = clean[i: i + batch_size]
                toks, attn = collate_batch(batch, self.tokenizer, max_len, device)
                z = self.model.sentence_embedding(toks, attn)
                outs.append(z.detach().cpu().numpy().astype("float32"))
            return np.concatenate(outs, axis=0)
        finally:
            if moved:
                self.model.to(self.device)
