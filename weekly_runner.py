# weekly_runner.py
import time, traceback, subprocess, sys

CMD = [sys.executable, "main.py"]

def run_once():
    print("🚀 Weekly learn cycle starting...")
    try:
        subprocess.run(CMD, check=True)
    except Exception:
        traceback.print_exc()
    print("✅ Weekly learn cycle finished.")

if __name__ == "__main__":
    while True:
        run_once()
        # Sleep ~7 days
        time.sleep(7 * 24 * 60 * 60)
