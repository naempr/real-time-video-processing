import os
import cv2
import yaml
import pandas as pd
import numpy as np
from datetime import timedelta

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def frame_to_timestamp(frame_idx, fps):
    secs = frame_idx / fps if fps > 0 else 0
    return str(timedelta(seconds=secs))

def save_csv(rows, path):
    ensure_dir_for_file(path)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def draw_polygon(frame, polygon, color=(0, 255, 255), thickness=2):
    # polygon: [[x1,y1], [x2,y2], ...]
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

def put_text(frame, text, org, color=(50, 255, 50), scale=0.8, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
