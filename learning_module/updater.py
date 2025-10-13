"""
Updater Module
Runs daily background updates (cron-like loop).
"""

import schedule
import time
from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer
from config_loader import Config

class Updater:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.safety = SafetyFilter()
        self.config = Config()
        self.trainer = Trainer()

    def update_once(self):
        print("🔄 Running scheduled update...")
        kb = KnowledgeBase()

        raw_data = self.fetcher.fetch()
        safe_data = self.safety.filter(raw_data)
        kb.store(safe_data)

        all_knowledge = kb.query("")
        if all_knowledge:
            epochs = self.config.get_nested("learning", "epochs", default=5)
            self.trainer.model.train(all_knowledge, [1]*len(all_knowledge), epochs=epochs)
            print("✅ Model retrained with latest knowledge")

        kb.close()

    def run_scheduler(self):
        update_time = self.config.get("update_time", "09:00")
        print(f"⏰ Scheduling daily updates at {update_time}")

        schedule.every().day.at(update_time).do(self.update_once)

        while True:
            schedule.run_pending()
            time.sleep(30)
