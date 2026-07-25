# pick_homography_points.py
import cv2
import argparse
import yaml
import os

points = []

def mouse_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([int(x), int(y)])
            print(f"[{len(points)}] click: ({x}, {y})")
        else:
            print("Already have 4 points. Press 'u' to undo or 'r' to reset.")

def draw_overlay(img, pts):
    out = img.copy()

    # draw selected points
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            out, str(i + 1), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

    # draw polyline
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), (255, 200, 0), 2)

    # close polygon if 4 points
    if len(pts) == 4:
        cv2.line(out, tuple(pts[3]), tuple(pts[0]), (255, 200, 0), 2)

    # helper text
    cv2.putText(
        out,
        "Click 4 points on ground: BL, BR, TR, TL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 255, 50),
        2,
    )
    cv2.putText(
        out,
        "r=reset | u=undo | q/esc=quit | auto-save at 4th point",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2,
    )
    cv2.putText(
        out,
        f"Points: {len(pts)}/4",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="path to mp4/video")
    parser.add_argument("--frame", type=int, default=0, help="frame index to pick points from")
    parser.add_argument("--width", type=float, default=12.0, help="ground rectangle width (meters)")
    parser.add_argument("--height", type=float, default=8.0, help="ground rectangle height (meters)")
    parser.add_argument("--out", default="", help="optional yaml output path")
    args = parser.parse_args()

    # read frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {args.frame}")

    # window + callback
    win = "Pick 4 ground points"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    print("\nInstructions:")
    print(" - روی زمین 4 نقطه بزن (ترتیب: پایین-چپ، پایین-راست، بالا-راست، بالا-چپ)")
    print(" - بعد از نقطه 4، برنامه خودکار تایید می‌کند و YAML می‌سازد")
    print(" - r: ریست | u: حذف آخرین نقطه | q/Esc: خروج")

    # interaction loop
    while True:
        vis = draw_overlay(frame, points)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20)

        # auto-finish when 4 points selected
        if len(points) == 4:
            print("Accepted 4 points (auto-finish).")
            break

        if key == -1:
            continue

        key8 = key & 0xFF

        # quit
        if key8 in (27, ord('q')):  # ESC / q
            print("Canceled.")
            cv2.destroyAllWindows()
            return

        # reset
        elif key8 == ord('r'):
            points.clear()
            print("Reset.")

        # undo
        elif key8 == ord('u'):
            if points:
                p = points.pop()
                print(f"Undo: {p}")

    cv2.destroyAllWindows()

    # corresponding ground points (rectangle in XY)
    # order must match image points: BL, BR, TR, TL
    ground_points = [
        [0.0, args.height],         # BL
        [args.width, args.height],  # BR
        [args.width, 0.0],          # TR
        [0.0, 0.0],                 # TL
    ]

    snippet = {
        "use_ground_plane": True,
        "homography": {
            "image_points": points,
            "ground_points": ground_points,
        }
    }

    print("\n=== YAML snippet ===")
    print(yaml.dump(snippet, sort_keys=False, allow_unicode=True))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.dump(snippet, f, sort_keys=False, allow_unicode=True)
        print(f"Saved: {os.path.abspath(args.out)}")
    else:
        print("No --out path provided; snippet printed only.")

if __name__ == "__main__":
    main()
