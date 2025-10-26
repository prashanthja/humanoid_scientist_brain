"""
Module: Safety Filter
Rewritten to remove TensorFlow/Keras dependencies.
Performs lightweight, context-aware text screening.
"""

import re


class SafetyFilter:
    def __init__(self):
        # Basic unsafe keyword list (can be expanded)
        self.blocklist = {
            "kill", "suicide", "harm", "weapon", "attack",
            "murder", "bomb", "terror", "violence",
            "blood", "drugs", "hate", "shoot", "explode",
        }

        # Whitelisted scientific contexts where some words are safe
        self.whitelist_patterns = [
            r"bomb calorimeter",        # scientific instrument
            r"reaction mechanism",      # chemistry context
            r"violence in video games", # social science
            r"drug development",        # pharma research
            r"quantum",                 # generic science context
        ]

    def is_safe(self, text: str) -> bool:
        """Check whether a text is safe using simple rules."""
        if not text or not isinstance(text, str):
            return True

        lower = text.lower()

        # Whitelisted scientific phrases override blocklist
        for pattern in self.whitelist_patterns:
            if re.search(pattern, lower):
                return True

        # Check for unsafe keywords
        for bad in self.blocklist:
            if bad in lower:
                return False

        return True

    def filter(self, data_list):
        """
        Filters out unsafe text entries from a list of strings or dicts.
        Returns only safe items.
        """
        safe = []
        for item in data_list:
            text = item.get("text") if isinstance(item, dict) else str(item)
            if self.is_safe(text):
                safe.append(item)
        return safe
