import os, json, hashlib

class TrainingMemory:
    def __init__(self, path="data/trained_papers.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.trained_hashes = set()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.trained_hashes = set(json.load(f))
            except Exception:
                self.trained_hashes = set()

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.trained_hashes)), f, indent=2)

    def _hash_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def mark_trained(self, items):
        for it in items:
            text = it.get("text", "") if isinstance(it, dict) else str(it)
            if text.strip():
                self.trained_hashes.add(self._hash_text(text))
        self._save()

    def filter_new(self, items):
        """Return only items not seen before"""
        new_items = []
        for it in items:
            text = it.get("text", "") if isinstance(it, dict) else str(it)
            h = self._hash_text(text)
            if h not in self.trained_hashes:
                new_items.append(it)
        return new_items
