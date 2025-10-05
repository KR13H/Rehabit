import json
import time

class SessionStore:
    def __init__(self):
        self.history = []
        self.start_time = time.time()

    def update(self, metrics):
        self.history.append(metrics)

    def finish(self):
        summary = {
            "duration": time.time() - self.start_time,
            "history": self.history
        }
        with open(f"session_{int(self.start_time)}.json", "w") as f:
            json.dump(summary, f)
        return summary
