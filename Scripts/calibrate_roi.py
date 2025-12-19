from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from Action_Detection_SOP.roi import RoiPolygon, draw_roi, save_roi_json


Point = Tuple[int, int]


def _read_single_frame(*, image: Optional[str], video: Optional[str], webcam: Optional[int], rtsp: Optional[str]):
    if image is not None:
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return frame

    if video is not None:
        cap = cv2.VideoCapture(video)
    elif rtsp is not None:
        cap = cv2.VideoCapture(rtsp)
    else:
        cap = cv2.VideoCapture(0 if webcam is None else int(webcam))

    if not cap.isOpened():
        raise RuntimeError("Could not open capture source.")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame from the source.")
        return frame
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalbrasi Interaktif ROI untuk kalibrasi polygonInteractive ROI polygon calibration (click points, save JSON).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default=None, help="Path to an input image.")
    src.add_argument("--video", default=None, help="Path to an input video file.")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0).")
    src.add_argument("--rtsp", default=None, help="RTSP URL (e.g., rtsp://user:pass@host/...).")
    parser.add_argument("--out", default="configs/roi.json", help="Output ROI JSON path.")
    args = parser.parse_args()

    frame = _read_single_frame(image=args.image, video=args.video, webcam=args.webcam, rtsp=args.rtsp)
    win = "ROI calibration"
    points: List[Point] = []
    saved = False

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    instructions = "L-click add | R-click undo | r reset | s save | q/ESC quit"

    try:
        while True:
            vis = frame.copy()
            if len(points) >= 3:
                vis = draw_roi(vis, RoiPolygon(points=tuple(points)))
            else:
                for (x, y) in points:
                    cv2.circle(vis, (x, y), radius=4, color=(0, 255, 255), thickness=-1)

            cv2.putText(vis, instructions, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, f"points={len(points)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, vis)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                points.clear()
                continue
            if key == ord("s"):
                if len(points) < 3:
                    print("ROI needs at least 3 points.")
                    continue
                out_path = Path(args.out)
                save_roi_json(out_path, RoiPolygon(points=tuple(points)))
                print(f"Saved ROI to: {out_path}")
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
