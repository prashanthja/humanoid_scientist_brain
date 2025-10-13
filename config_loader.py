"""
Config Loader
Loads settings from config.json
"""

import json
import os

class Config:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return json.load(f)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def get_nested(self, *keys, default=None):
        """
        Example: get_nested("learning", "epochs")
        """
        d = self.data
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d
