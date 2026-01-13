"""
calibrate_roi.py
Merupakan program yang digunakan untuk mengkalibrasi area *Region of Interset* (ROI)
dari area deteksi yang diinginkan
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from Action_Detection_SOP.roi import RoiPolygon, draw_roi, save_roi_json


Point = Tuple[int, int]


def _read_single_frame(
    *, image: Optional[str], video: Optional[str], webcam: Optional[int], rtsp: Optional[str]
) -> "cv2.typing.MatLike":
    """Read a single frame from exactly one source."""
    if image is not None:
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return frame

    cap: Optional[cv2.VideoCapture]
    if video is not None:
        cap = cv2.VideoCapture(video)
    elif rtsp is not None:
        cap = cv2.VideoCapture(rtsp)
    elif webcam is not None:
        cap = cv2.VideoCapture(int(webcam))
    else:  # should be unreachable due to argparse, but keep it safe for direct calls
        raise ValueError("Invalid input source: provide exactly one of image/video/webcam/rtsp.")

    if not cap.isOpened():
        src = video if video is not None else (rtsp if rtsp is not None else f"webcam:{webcam}")
        raise RuntimeError(f"Tidak bisa membuka sumber video capture: {src}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Tidak bisa membaca frame dari source yang ada")
        return frame
    
    finally:
        cap.release()



def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive ROI polygon untuk kalibrasi (menentukan area yang ingin dijadikan ROI) save di JSON")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", default=None, help="path ke input image")
    src.add_argument("--video", default=None, help="path ke input video")
    src.add_argument("--webcam", type=int, default=None, help="Webcam index")
    src.add_argument("--rtsp", default=None, help="URL RTSP (r.g., rtsp://user:pass@host/...) etc, etc")
    parser.add_argument("--out", default="configs/roi.json", help="Output ROI untuk json.")
    args = parser.parse_args()

    frame = _read_single_frame(image=args.image, video=args.video, webcam=args.webcam, rtsp=args.rtsp)
    frame_h, frame_w = frame.shape[:2]
    win = "ROI Calibration"
    points: List[Point] = []
    saved = False

    def on_mouse(event: int, x: int, y:int, flgas:int, param:object ) -> None:
        """fungsi yang berfungsi untuk membuat kotaknya"""
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    instructions = "L-click add | R-click undo | r reset | s save | q/ESC untuk keluar"

    try:
        while True:
            vis = frame.copy()
            if len(points) >=3:
                vis = draw_roi(vis, RoiPolygon(points=tuple(points)))
            else:
                for (x, y) in points:
                    cv2.circle(vis, (x, y), radius=4, color=(0,255, 255), thickness=-1)
                    #put text to explain
            cv2.putText(vis, instructions, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))
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
                    print("ROI harus memilik 3 titik untuk berjalan dengan baik")
                    continue
                out_path = Path(args.out)
                save_roi_json(out_path, RoiPolygon(points=tuple(points), frame_size=(frame_w, frame_h)))
                print(f"Menyimpan ROI di {out_path}")
                saved = True
                break
    finally:
        cv2.destroyAllWindows()

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
