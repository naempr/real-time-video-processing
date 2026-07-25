

import time
import cv2
import numpy as np

from utils import (
    load_config,
    ensure_dir_for_file,
    frame_to_timestamp,
    save_csv,
    draw_polygon,
    put_text,
)
from detector_tracker import DetectorTracker
from roi import ROIManager
from event_engine import EventEngine
from metrics import PerfMeter, dump_summary
from homography import HomographyProjector


def _project_ground_polygon_to_image(projector, ground_polygon):
    pts_img = []
    for gx, gy in ground_polygon:
        ix, iy = projector.ground_to_image(float(gx), float(gy))
        pts_img.append([float(ix), float(iy)])
    return pts_img


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)

    cap = cv2.VideoCapture(cfg["input_video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg['input_video']}")

   
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
   
    fps_input_reported = float(raw_fps) if raw_fps and raw_fps > 0 else None

   
    fallback_fps = float(cfg.get("fallback_fps", 25.0))

   
    if fps_input_reported is not None:
        fps_output_used = fps_input_reported
        fps_input_is_fallback = False
    else:
        fps_output_used = fallback_fps
        fps_input_is_fallback = True

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"[INFO] Video resolution = {w} x {h}")

    
    ensure_dir_for_file(cfg["output_video"])
    ensure_dir_for_file(cfg["events_csv"])
    ensure_dir_for_file(cfg["occupancy_csv"])
    ensure_dir_for_file(cfg["summary_json"])

    writer = cv2.VideoWriter(
        cfg["output_video"],
        cv2.VideoWriter_fourcc(*"mp4v"),  
        fps_output_used,
        (w, h),
    )

    dt = DetectorTracker(
        model_path=cfg["model_path"],
        tracker_yaml=cfg["tracker_yaml"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        imgsz=cfg["imgsz"],
        classes=cfg["classes"],
    )

    # ---------------------------
    # Mode: IMAGE vs GROUND
    # ---------------------------
    use_ground_plane = bool(cfg.get("use_ground_plane", False))
    projector = None
    roi_projected_draw = None

    roi_draw_image = cfg["polygon"]

    if use_ground_plane:
        hcfg = cfg.get("homography", {})
        image_points = hcfg.get("image_points", [])
        ground_points = hcfg.get("ground_points", [])

        if len(image_points) != 4 or len(ground_points) != 4:
            raise ValueError(
                "For use_ground_plane=True, homography.image_points and homography.ground_points must each have exactly 4 points."
            )

        projector = HomographyProjector.from_points(image_points, ground_points)

        ground_polygon = cfg.get("ground_polygon", None)
        if not ground_polygon or len(ground_polygon) < 3:
            raise ValueError(
                "For use_ground_plane=True, provide ground_polygon with at least 3 points."
            )

        roi_logic = ROIManager(ground_polygon)
        roi_projected_draw = _project_ground_polygon_to_image(projector, ground_polygon)
    else:
        roi_logic = ROIManager(cfg["polygon"])

    engine = EventEngine(
        inside_req=cfg["inside_streak_required"],
        outside_req=cfg["outside_streak_required"],
        min_confirm_frames=cfg["min_confirm_frames"],
        max_missed_inside=cfg["max_missed_frames_for_inside"],
    )

    meter = PerfMeter()
    events_rows = []
    occ_rows = []

    line_thickness = int(cfg.get("line_thickness", 2))
    font_scale = float(cfg.get("font_scale", 0.8))

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t0 = time.time()

        res = dt.infer(frame)

        dets = []
        if (
            res is not None
            and res.boxes is not None
            and res.boxes.id is not None
            and len(res.boxes.id) > 0
        ):
            ids = res.boxes.id.int().cpu().tolist()
            xyxy = res.boxes.xyxy.cpu().tolist()
            for tid, b in zip(ids, xyxy):
                x1, y1, x2, y2 = b
                dets.append(
                    {
                        "track_id": int(tid),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        for d in dets:
            tid = int(d["track_id"])
            x1, y1, x2, y2 = d["bbox"]

            cx = 0.5 * (x1 + x2)
            cy = y2

            if use_ground_plane:
                gx, gy = projector.image_to_ground(cx, cy)
                inside_now = roi_logic.is_inside(gx, gy)
                lx, ly = float(gx), float(gy)
            else:
                inside_now = roi_logic.is_inside(cx, cy)
                lx, ly = float(cx), float(cy)

            ev = engine.update_track(tid, frame_idx, bool(inside_now))

            color = (0, 200, 0) if inside_now else (0, 120, 255)
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                line_thickness,
            )
            cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)

            put_text(
                frame,
                f"ID:{tid} {'IN' if inside_now else 'OUT'}",
                (int(x1), max(20, int(y1) - 8)),
                color=color,
                scale=0.6,
                thickness=2,
            )

            if ev is not None:
                occ_now = engine.current_occupancy()
                events_rows.append(
                    {
                        "timestamp": frame_to_timestamp(frame_idx, fps_output_used),
                        "frame_idx": frame_idx,
                        "track_id": tid,
                        "event": ev,
                        "x": lx,
                        "y": ly,
                        "img_x": float(cx),
                        "img_y": float(cy),
                        "mode": "GROUND" if use_ground_plane else "IMAGE",
                        "occupancy_current": occ_now,
                    }
                )

        forced = engine.finalize_missed(frame_idx)
        for tid in forced:
            occ_now = engine.current_occupancy()
            events_rows.append(
                {
                    "timestamp": frame_to_timestamp(frame_idx, fps_output_used),
                    "frame_idx": frame_idx,
                    "track_id": int(tid),
                    "event": "EXIT_FORCED_MISSED",
                    "x": None,
                    "y": None,
                    "img_x": None,
                    "img_y": None,
                    "mode": "GROUND" if use_ground_plane else "IMAGE",
                    "occupancy_current": occ_now,
                }
            )

        occ = engine.current_occupancy()
        occ_rows.append(
            {
                "timestamp": frame_to_timestamp(frame_idx, fps_output_used),
                "frame_idx": frame_idx,
                "occupancy": occ,
            }
        )

        draw_polygon(
            frame,
            roi_draw_image,
            color=(0, 255, 255),
            thickness=line_thickness,
        )
        if roi_projected_draw is not None and len(roi_projected_draw) >= 3:
            pts = np.array(roi_projected_draw, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=(255, 0, 255),
                thickness=line_thickness,
            )

        put_text(
            frame,
            f"Occupancy: {occ}",
            (20, 40),
            scale=font_scale,
            thickness=2,
        )
        put_text(
            frame,
            f"Frame: {frame_idx}",
            (20, 75),
            scale=0.7,
            thickness=2,
        )

        latency_ms = (time.time() - t0) * 1000.0
        meter.update(latency_ms)
        put_text(
            frame,
            f"Latency: {latency_ms:.1f} ms",
            (20, 110),
            scale=0.7,
            thickness=2,
        )

        mode_txt = "GROUND" if use_ground_plane else "IMAGE"
        put_text(
            frame,
            f"Mode: {mode_txt} | IDs: TRACKER_ONLY",
            (20, 145),
            scale=0.65,
            thickness=2,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    save_csv(events_rows, cfg["events_csv"])
    save_csv(occ_rows, cfg["occupancy_csv"])

    summary = {
        "perf": meter.summary(),  
        "final_occupancy": int(engine.current_occupancy()),
        "total_events": int(len(events_rows)),
        "use_ground_plane": bool(use_ground_plane),
        "use_id_reconciler": False,
       
        "fps_input_reported": fps_input_reported,
        "fps_output_used": float(fps_output_used),
        "fallback_fps": float(fallback_fps),
        "fps_input_is_fallback": bool(fps_input_is_fallback),
    }
    dump_summary(cfg["summary_json"], summary)

    print("Done.")
    print(summary)


if __name__ == "__main__":
    run()
