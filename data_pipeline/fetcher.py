# data_pipeline/fetcher.py

import requests
import random

class DataFetcher:
    """
    DataFetcher retrieves raw information about a given topic.
    In this version, it simulates web/API fetching.
    """

    def __init__(self):
        # You can later extend this with API keys, base URLs, etc.
        self.sources = [
            "https://example.com/science",
            "https://example.com/physics",
            "https://example.com/mathematics",
            "https://example.com/ai"
        ]

    def fetch(self, topic: str) -> str:
        """
        Fetch raw data related to a specific topic.
        Args:
            topic (str): The subject the robot wants to learn.
        Returns:
            str: Raw textual data about the topic.
        """
        print(f"[Fetcher] Fetching new data for topic: {topic}")

        # For now we simulate fetching by combining topic with a random source
        fake_source = random.choice(self.sources)
        fake_data = (
            f"This is simulated raw data about {topic}. "
            f"Source: {fake_source}. "
            f"The data contains references to {topic} concepts, theories, and discussions."
        )

        return fake_data


if __name__ == "__main__":
    # Quick test
    fetcher = DataFetcher()
    result = fetcher.fetch("quantum mechanics")
    print(result)
