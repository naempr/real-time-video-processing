import time
import json
import numpy as np

class PerfMeter:
    def __init__(self):
        self.latencies_ms = []
        self.frame_count = 0
        self.t0 = time.time()

    def update(self, latency_ms):
        self.latencies_ms.append(latency_ms)
        self.frame_count += 1

    def summary(self):
        elapsed = time.time() - self.t0
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0.0
        arr = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {
            "frames": int(self.frame_count),
            "fps_avg": float(round(fps_avg, 3)),
            "latency_ms_avg": float(round(arr.mean(), 3)),
            "latency_ms_p95": float(round(np.percentile(arr, 95), 3)),
            "latency_ms_max": float(round(arr.max(), 3))
        }

def dump_summary(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
