import os, json, hashlib

class TrainingMemory:
    """
    TrainingMemory — Prevents duplicate training and handles None text safely.
    Keeps track of already-trained papers by storing SHA-1 hashes of their text.
    """

    def __init__(self, path="data/trained_papers.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.trained_hashes = set()
        self._load()

    # --------------------------------------------------------
    def _load(self):
        """Load previously trained hashes from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.trained_hashes = set(data if isinstance(data, list) else [])
                print(f"💾 Loaded {len(self.trained_hashes)} trained paper hashes.")
            except Exception as e:
                print(f"⚠️ Failed to load training memory: {e}")
                self.trained_hashes = set()
        else:
            print("🧠 Training memory initialized empty.")

    def _save(self):
        """Persist the trained hashes to disk."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(self.trained_hashes)), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save training memory: {e}")

    def _hash_text(self, text) -> str:
        """Compute a safe hash for any text (handles None or non-string)."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    # --------------------------------------------------------
    def mark_trained(self, items):
        """Record items as trained."""
        added = 0
        for it in items:
            text = it.get("text") if isinstance(it, dict) else str(it)
            if not text or not str(text).strip():
                continue
            h = self._hash_text(text)
            if h not in self.trained_hashes:
                self.trained_hashes.add(h)
                added += 1
        self._save()
        print(f"💾 Marked {added} new items as trained (total={len(self.trained_hashes)}).")

    def filter_new(self, items):
        """Return only items that are not already trained."""
        new_items = []
        skipped = 0
        for it in items:
            text = it.get("text") if isinstance(it, dict) else str(it)
            if not text or not str(text).strip():
                skipped += 1
                continue
            h = self._hash_text(text)
            if h not in self.trained_hashes:
                new_items.append(it)
            else:
                skipped += 1
        print(f"🧩 TrainingMemory: {len(new_items)} new, {skipped} skipped.")
        return new_items
