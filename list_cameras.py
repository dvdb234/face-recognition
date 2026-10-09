#!/usr/bin/env python3
"""
Scan camera indices and report which ones give a REAL (non-black) image.

Run this when preview.py shows a black window, to find the right --camera index:

    python list_cameras.py

For each index 0..5 it opens the camera, grabs several frames (to let auto-
exposure warm up), and reports the resolution and average brightness. A bright
value (well above ~5) means a real picture; a value near 0 means black frames
(often a disabled/virtual/IR camera, a closed privacy shutter, or a Windows
camera-privacy block).
"""

import sys
import cv2

# Per-OS backend order. Windows prefers MSMF (DirectShow can report a black
# image when auto-exposure is off); Linux/Raspberry Pi USB webcams use V4L2.
if sys.platform.startswith("win"):
    BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
elif sys.platform.startswith("linux"):
    BACKENDS = [("V4L2", cv2.CAP_V4L2), ("default", cv2.CAP_ANY)]
else:
    BACKENDS = [("default", cv2.CAP_ANY)]

WARMUP_FRAMES = 10   # discard/allow a few frames for the sensor to wake up


def probe(index):
    """Return a human-readable result string for one camera index."""
    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        # DirectShow may open with auto-exposure off (black image) -- enable it.
        if backend == cv2.CAP_DSHOW:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        brightness = None
        w = h = 0
        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                brightness = float(frame.mean())
        cap.release()

        if brightness is None:
            return f"index {index}: opened ({name}) but returned NO frames"

        verdict = "REAL IMAGE" if brightness > 5 else "black/blank"
        return (f"index {index}: {w}x{h}  avg brightness {brightness:5.1f}  "
                f"[{verdict}]  (backend {name})")

    return f"index {index}: no camera"


def main():
    print("Scanning camera indices 0..5 (this takes a few seconds)...\n")
    good = []
    for i in range(6):
        result = probe(i)
        print("  " + result)
        if "REAL IMAGE" in result:
            good.append(i)

    print()
    if good:
        print(f"Use one of these that shows YOU: --camera {good[0]}")
        if len(good) > 1:
            print(f"(other working indices: {good})")
        print(f"\n    python preview.py --camera {good[0]}")
    else:
        print("No index produced a real image. Likely causes on Windows:")
        print("  1. Camera privacy: Settings > Privacy & security > Camera >")
        print("     turn ON 'Camera access' AND 'Let desktop apps access your camera'.")
        print("  2. A physical privacy shutter/slider is covering the lens.")
        print("  3. Another app (Zoom/Teams/browser) is holding the camera -- close it.")
        print("  4. A laptop IR/Windows Hello camera with no visible-light output.")


if __name__ == "__main__":
    main()
