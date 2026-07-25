# FenceWatch: Real-Time Virtual Fence Occupancy and Event Pipeline

FenceWatch is a computer vision pipeline for real-time people counting, occupancy monitoring, and entry/exit event generation inside a virtual fence region. It uses **YOLO** for person detection and **BoT-SORT** for multi-object tracking, then evaluates each tracked person against a user-defined polygonal **region of interest (ROI)** to determine whether they are inside or outside the monitored area.

The system is designed for smart surveillance, virtual perimeter monitoring, and occupancy analytics.

---

## Project Structure

```text
FenceWatch/
├── configs/
│   ├── config.yaml
│   └── botsort_reid.yaml
├── src/
│   ├── main.py
│   ├── detector_tracker.py
│   ├── event_engine.py
│   ├── roi.py
│   ├── homography.py
│   ├── metrics.py
│   └── utils.py
├── pick_homography_points.py
├── homography_snippet.yaml
├── README.md
└── requirements.txt
```

---

## Component Overview

### `src/main.py`
The main entry point of the pipeline. It loads configuration, opens the input video, runs detection and tracking, evaluates ROI membership, updates the event engine, renders overlays when enabled, and writes output artifacts.

### `src/detector_tracker.py`
Wraps **YOLO** detection and **BoT-SORT** tracking into a single module. It provides bounding boxes and stable track IDs for detected people across frames.

### `src/event_engine.py`
Implements event logic and state transitions. It converts noisy frame-level inside/outside observations into stable `ENTER` and `EXIT` events, maintains occupancy count, and handles missed tracks.

### `src/roi.py`
Contains polygon-based ROI evaluation logic. It determines whether a reference point for each tracked person, typically the bottom-center of the bounding box, lies inside the virtual fence.

### `src/homography.py`
Provides utilities for mapping image coordinates onto a ground plane using homography. This functionality is available in the codebase but is not enabled in the default pipeline run.

### `src/metrics.py`
Collects runtime performance data and produces statistics such as processing rate, latency summaries, event totals, and final occupancy.

### `src/utils.py`
Includes shared helper functions for configuration loading, path handling, drawing, and miscellaneous utilities.

---


## Runtime Configuration

Runtime parameters are defined in:

```text
configs/config.yaml
```

This file controls:
- input video path
- ROI polygon
- output file paths
- detector settings
- tracker configuration (via `configs/botsort_reid.yaml`)
- visualization options
- event confirmation behavior

The BoT-SORT tracker used by the pipeline is configured through the file:

```text
configs/botsort_reid.yaml
```

This configuration file defines tracking parameters such as association thresholds, ReID settings, and other tracker-specific behavior used by the YOLO `track()` interface.

> **Note:** The ROI polygon was rescaled from **2560×1440** to **1920×1080**, assuming the original coordinates provided by the client were defined on a higher-resolution reference frame and exceeded the bounds of the current video.


---

## Output Paths

The current implementation writes outputs directly into the `outputs/` directory:

```yaml
output_video: "outputs/annotated.mp4"
events_csv: "outputs/events.csv"
occupancy_csv: "outputs/occupancy.csv"
summary_json: "outputs/summary.json"
```


## Stability Design

FenceWatch is designed to reduce flicker and unstable event generation in real scenes.

- **Frame-persistence confirmation**  
  Entry and exit decisions are confirmed only after consistent observations across multiple consecutive frames.

- **Track-ID-based temporal consistency**  
  The pipeline relies on tracker-generated identities rather than raw detection row order, providing better temporal stability.

- **State-transition event generation**  
  Events are emitted from confirmed inside/outside state changes, not from single-frame decisions.

This design helps reduce false transitions near polygon boundaries and improves occupancy stability.

To further improve track continuity and reduce ID switches, stronger YOLO variants can be used when the current detector is weak for the scene. In addition, an appearance embedding can be built for each `track_id` and used as an extra identity cue to improve matching consistency during occlusion, temporary misses, and re-association.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the pipeline from the project root:

```bash
python src/main.py
```

Before running, update `configs/config.yaml` as needed for your input video, ROI, and output settings.

The test video is expected to be placed in the project root directory with the following name:

```text
input.mp4
```

---

## Output Files

### `outputs/events.csv`
Stores confirmed event records generated by the event engine.

Typical fields include:
- `timestamp`
- `frame_idx`
- `track_id`
- `event_type`

Typical event types:
- `ENTER`
- `EXIT`
- `EXIT_FORCED_MISSED`

### `outputs/occupancy.csv`
Stores frame-by-frame occupancy values for the monitored region.

Typical fields include:
- `frame_idx`
- `occupancy_count`

### `outputs/summary.json`
Stores run-level summary information, including:
- processing statistics
- latency metrics
- event totals
- final occupancy
- a snapshot of runtime configuration

---

## Performance Benchmark

Example benchmark results from a sample run:

| Metric | Value |
| :--- | :--- |
| Total processed frames | 2,086 |
| Average throughput | 21.819 FPS |
| Mean latency | 30.463 ms/frame |
| P95 latency | 31.839 ms/frame |
| Max latency | 4,312.853 ms |
| Total emitted events | 60 |
| Final occupancy | 0 |

### Interpretation
- The pipeline processes close to the input video rate of **25 FPS**, achieving about **21.8 FPS** on the tested hardware.
- Mean latency and P95 latency are very close (~30–32 ms), indicating stable processing time for most frames.
- The maximum latency spike is significantly larger and is typically caused by initialization overhead, model warm-up, or temporary system resource contention.
- Event generation remained stable during the run, producing **60 confirmed ENTER/EXIT events**, with a final monitored occupancy of **0**.

---

## Optional Homography Point Selection
Run the following command to interactively select the corresponding homography points from a video frame, then add the generated output to config.yaml:


```bash
python pick_homography_points.py --video input.mp4 --frame 300 --out homography_snippet.yaml
```

---

## Summary


This implementation is an initial demo of the overall system, intended to demonstrate the core online pipeline, virtual fence occupancy logic, and entry/exit event generation. However, it requires further improvements.