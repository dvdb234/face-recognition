#!/usr/bin/env python3
"""
Minimal webcam viewer -- just shows what the camera sees.

Use this to confirm your USB webcam works (and find the right --camera index)
BEFORE installing the heavy detection stack. It only needs OpenCV:

    pip install opencv-python
    python preview.py

Press 'q' (with the window focused) or close the window to quit.
If the picture is wrong/black, try another camera: --camera 1, --camera 2, ...
"""

import argparse
import sys
import time

import cv2


def _backend_list(choice):
    """Map a --backend choice to the OpenCV backends to try, in order."""
    if choice == "dshow":
        return [cv2.CAP_DSHOW]
    if choice == "msmf":
        return [cv2.CAP_MSMF]
    if choice == "v4l2":
        return [cv2.CAP_V4L2]
    if choice == "any":
        return [cv2.CAP_ANY]
    # "auto": per-OS. Windows prefers MSMF (DirectShow can hand a black frame
    # when auto-exposure is off); Linux/Raspberry Pi USB webcams use V4L2.
    if sys.platform.startswith("win"):
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(index, width, height, backend_choice="auto", mjpg=False):
    """Open a webcam, or exit with a helpful message. (Same logic as recognize.py.)"""
    backends = _backend_list(backend_choice)

    cap = None
    used = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            used = backend
            break

    if cap is None or not cap.isOpened():
        print(
            f"ERROR: could not open camera index {index}.\n"
            "       - Is a USB webcam plugged in?\n"
            "       - Is another app (Zoom, Teams, browser) using it? Close it.\n"
            "       - Try a different index: --camera 1, --camera 2, ...",
            file=sys.stderr,
        )
        sys.exit(1)

    # DirectShow can open some cameras with auto-exposure disabled, which yields
    # a fully black image. Explicitly enable auto-exposure (0.75 = auto in the
    # DirectShow convention). MSMF handles exposure correctly on its own.
    if used == cv2.CAP_DSHOW:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # MJPG lets USB webcams (esp. on the Pi) reach full FPS instead of being
    # stuck at a few FPS on uncompressed YUYV. Set before the resolution.
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, _ = cap.read()
    if not ok:
        print(
            f"ERROR: camera {index} opened but returned no frames. "
            "Try another --camera index or replug the webcam.",
            file=sys.stderr,
        )
        cap.release()
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {index} opened at {w}x{h}. Press 'q' to quit.")
    return cap


def main():
    p = argparse.ArgumentParser(description="Show the raw webcam feed.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default 0).")
    p.add_argument("--width", type=int, default=640, help="Requested width.")
    p.add_argument("--height", type=int, default=480, help="Requested height.")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2", "any"],
                   default="auto",
                   help="Camera backend. 'auto' picks per-OS. On Windows, try "
                        "'msmf' if the image is dark/black.")
    p.add_argument("--mjpg", action="store_true",
                   help="Request MJPG format (recommended for USB webcams on "
                        "the Raspberry Pi to reach full FPS).")
    args = p.parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.backend, args.mjpg)
    window = "Webcam preview  (press q to quit)"

    fps = 0.0
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: dropped a frame.", file=sys.stderr)
                continue

            # Rolling FPS estimate.
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            cv2.putText(frame, f"cam {args.camera}  {fps:4.1f} FPS", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            # Stop if the user closed the window with the [x] button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Bye.")


if __name__ == "__main__":
    main()
