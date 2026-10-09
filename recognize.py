#!/usr/bin/env python3
"""
Live object + face recognition on a USB webcam.

The same file runs on a desktop/laptop (Windows/macOS/Linux) for development
and on a Raspberry Pi 5 for deployment. Because both use a plain USB webcam,
nothing platform-specific is needed in the recognition code itself.

Pipeline
--------
1. Grab a frame from the webcam (OpenCV).
2. Every N frames, run YOLOv8n to detect objects (boxes + labels).
3. Every M frames, run face_recognition to find and identify known faces.
4. Cache both results and redraw them on *every* frame so the video stays
   smooth even though the heavy models only run occasionally.

The two "every N frames" knobs are what keep this usable on a Raspberry Pi 5
CPU: running full YOLO + face recognition on every single frame would tank the
frame rate, so we run them periodically and reuse the last result in between.

Run `python recognize.py --help` for all options.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Optional heavy dependencies are imported defensively so that a missing or
# half-installed package produces a clear message instead of a raw traceback.
# ----------------------------------------------------------------------------

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "ERROR: the 'ultralytics' package is not installed.\n"
        "       Install the project requirements first:\n"
        "           pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)

# face_recognition (and its dlib backend) is the trickiest dependency to build,
# especially on a Raspberry Pi. If it isn't available we still run object
# detection so you always have *something* working -- face labels are simply
# skipped until the library is installed.
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


# ----------------------------------------------------------------------------
# Colors (BGR, because OpenCV) used for drawing.
# ----------------------------------------------------------------------------
COLOR_OBJECT = (0, 200, 0)      # green boxes for YOLO objects
COLOR_KNOWN_FACE = (255, 128, 0)  # blue-ish for recognised people
COLOR_UNKNOWN_FACE = (0, 0, 255)  # red for unknown people
COLOR_TEXT = (255, 255, 255)    # white label text
COLOR_HUD = (0, 255, 255)       # yellow for the on-screen stats


# ============================================================================
# Known-face database
# ============================================================================
class KnownFaces:
    """Loads reference photos from a folder and holds their face encodings.

    Each image file becomes one known person. The person's name is derived
    from the file name, so ``faces/david.jpg`` -> "David" and
    ``faces/alex_smith.png`` -> "Alex Smith".
    """

    def __init__(self):
        self.encodings = []  # list of 128-d numpy vectors
        self.names = []      # parallel list of display names

    @staticmethod
    def _name_from_filename(path: Path) -> str:
        """faces/alex_smith.jpg -> 'Alex Smith'."""
        stem = path.stem.replace("_", " ").replace("-", " ")
        return stem.strip().title()

    def load(self, folder: str) -> None:
        """Populate encodings/names from every image in *folder*.

        Missing folder or no usable photos are handled gracefully: we print a
        warning and carry on. Any detected face will then simply be labelled
        "Unknown" until reference photos are added.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return

        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(
                f"WARNING: faces folder '{folder}' does not exist yet.\n"
                f"         Create it and drop in reference photos "
                f"(e.g. {folder}/david.jpg). All faces will be labelled "
                f"'Unknown' until then.",
                file=sys.stderr,
            )
            return

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in image_exts
        )

        if not image_files:
            print(
                f"WARNING: no reference photos found in '{folder}'.\n"
                f"         Add images like {folder}/david.jpg to recognise "
                f"people. Faces will be labelled 'Unknown' until then.",
                file=sys.stderr,
            )
            return

        for img_path in image_files:
            try:
                image = face_recognition.load_image_file(str(img_path))
                # A reference photo should contain exactly one clear face.
                encs = face_recognition.face_encodings(image)
            except Exception as exc:  # corrupt image, unreadable file, etc.
                print(f"WARNING: could not process '{img_path.name}': {exc}",
                      file=sys.stderr)
                continue

            if not encs:
                print(
                    f"WARNING: no face found in '{img_path.name}' -- skipping. "
                    f"Use a clear, front-facing photo.",
                    file=sys.stderr,
                )
                continue
            if len(encs) > 1:
                print(
                    f"WARNING: '{img_path.name}' has multiple faces; using the "
                    f"first one. A solo photo works best.",
                    file=sys.stderr,
                )

            name = self._name_from_filename(img_path)
            self.encodings.append(encs[0])
            self.names.append(name)
            print(f"  loaded reference face: {name}  ({img_path.name})")

        print(f"Known people: {len(self.names)}")

    def identify(self, encoding, tolerance: float) -> str:
        """Return the best-matching known name, or 'Unknown'.

        Uses face *distance* (lower = more similar) rather than the boolean
        compare so that, when several references are close, we pick the single
        best match.
        """
        if not self.encodings:
            return "Unknown"

        distances = face_recognition.face_distance(self.encodings, encoding)
        best_index = int(np.argmin(distances))
        if distances[best_index] <= tolerance:
            return self.names[best_index]
        return "Unknown"


# ============================================================================
# Drawing helpers
# ============================================================================
def draw_label(frame, x1, y1, text, box_color):
    """Draw *text* in a filled bar sitting just above (x1, y1)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Keep the label on-screen if the box is near the top edge.
    top = max(y1, th + baseline + 4)
    cv2.rectangle(
        frame,
        (x1, top - th - baseline - 4),
        (x1 + tw + 4, top),
        box_color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, text, (x1 + 2, top - baseline - 2),
        font, scale, COLOR_TEXT, thickness, cv2.LINE_AA,
    )


def draw_objects(frame, detections):
    """Draw cached YOLO detections: list of (x1, y1, x2, y2, label, conf)."""
    for x1, y1, x2, y2, label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OBJECT, 2)
        draw_label(frame, x1, y1, f"{label} {conf:.0%}", COLOR_OBJECT)


def draw_faces(frame, faces):
    """Draw cached faces: list of (left, top, right, bottom, name)."""
    for left, top, right, bottom, name in faces:
        color = COLOR_UNKNOWN_FACE if name == "Unknown" else COLOR_KNOWN_FACE
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        draw_label(frame, left, top, name, color)


def draw_hud(frame, fps, obj_count, face_count, faces_enabled):
    """Small stats overlay in the top-left corner."""
    faces_txt = f"faces:{face_count}" if faces_enabled else "faces:off"
    text = f"{fps:4.1f} FPS | objs:{obj_count} | {faces_txt}"
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, COLOR_HUD, 2, cv2.LINE_AA)


# ============================================================================
# Detection steps
# ============================================================================
def detect_objects(model, frame, conf, imgsz):
    """Run YOLO on one frame and return a list of drawable detections.

    imgsz is the size YOLO internally resizes to before inference. It is the
    biggest performance lever on a CPU: smaller = much faster, with only minor
    accuracy loss for a webcam. (640 is the default; 480/416/320 are faster.)
    """
    # verbose=False keeps ultralytics from printing a line per frame.
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names.get(cls, str(cls))
        detections.append((x1, y1, x2, y2, label, confidence))
    return detections


def recognize_faces(known: KnownFaces, frame, scale, fr_model, tolerance):
    """Detect + identify faces on a downscaled copy of *frame*.

    Downscaling is the single biggest speed win for face_recognition on a CPU:
    we look for faces on a small image, then scale the coordinates back up to
    the full-resolution frame for drawing.
    """
    # Resize down, then convert BGR (OpenCV) -> RGB (face_recognition).
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_small, model=fr_model)
    encodings = face_recognition.face_encodings(rgb_small, locations)

    inv = 1.0 / scale
    faces = []
    for (top, right, bottom, left), encoding in zip(locations, encodings):
        name = known.identify(encoding, tolerance)
        faces.append((
            int(left * inv), int(top * inv),
            int(right * inv), int(bottom * inv),
            name,
        ))
    return faces


# ============================================================================
# Camera helper
# ============================================================================
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
    # "auto":
    if sys.platform.startswith("win"):
        # Prefer MSMF -- some integrated cameras hand DirectShow a black frame
        # when auto-exposure is off. Fall back to DirectShow, then anything.
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        # Raspberry Pi / Linux: USB webcams are exposed through V4L2.
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(index, width, height, backend_choice="auto", mjpg=False):
    """Open a webcam, returning a VideoCapture or exiting with advice."""
    cap = None
    used = None
    for backend in _backend_list(backend_choice):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            used = backend
            break

    if cap is None or not cap.isOpened():
        print(
            f"ERROR: could not open camera index {index}.\n"
            "       - Is a USB webcam plugged in?\n"
            "       - Is another program (Zoom, browser, etc.) using it?\n"
            "       - Try a different index: --camera 1, --camera 2, ...\n"
            "       - On Linux/Raspberry Pi, check that /dev/video* exists and "
            "you have permission (the 'video' group).",
            file=sys.stderr,
        )
        sys.exit(1)

    # DirectShow can open some cameras with auto-exposure disabled, producing a
    # fully black image. Explicitly enable auto-exposure (0.75 = auto in the
    # DirectShow convention). MSMF handles exposure correctly on its own.
    if used == cv2.CAP_DSHOW:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # Request MJPG. Many USB webcams default to uncompressed YUYV, which the USB
    # bus can only sustain at a few FPS (especially on a Raspberry Pi). MJPG is
    # compressed on-camera, so it hits full frame rate. Set this BEFORE the
    # resolution. Harmless on cameras that don't support it (they ignore it).
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Request a capture resolution. Lower resolution = higher FPS, which
    # matters a lot on the Pi. The driver may pick the closest supported size.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Sanity-check that we can actually read a frame.
    ok, _ = cap.read()
    if not ok:
        print(
            f"ERROR: camera {index} opened but returned no frames. "
            "Try another --camera index or replug the webcam.",
            file=sys.stderr,
        )
        cap.release()
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {index} opened at {actual_w}x{actual_h}.")
    return cap


# ============================================================================
# Argument parsing
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Live object + face recognition on a USB webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Webcam index (0 is usually the first camera).")
    p.add_argument("--width", type=int, default=640,
                   help="Requested capture width.")
    p.add_argument("--height", type=int, default=480,
                   help="Requested capture height.")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2", "any"],
                   default="auto",
                   help="Camera backend. 'auto' picks per-OS (MSMF on Windows, "
                        "V4L2 on Linux/Pi). Use 'msmf' if the picture is black "
                        "on Windows.")
    p.add_argument("--mjpg", action="store_true",
                   help="Request MJPG capture format. Recommended for USB "
                        "webcams on the Raspberry Pi -- many are stuck at a few "
                        "FPS otherwise.")

    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights. yolov8n.pt (nano) is the fastest and is "
                        "downloaded automatically on first run.")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Minimum confidence for an object detection.")
    p.add_argument("--imgsz", type=int, default=480,
                   help="YOLO inference size. Biggest speed lever on CPU: "
                        "lower is much faster (e.g. 320), 640 is most accurate.")
    p.add_argument("--detect-every", type=int, default=2, metavar="N",
                   help="Run YOLO once every N frames (higher = faster, "
                        "laggier boxes). 1 = every frame.")

    p.add_argument("--faces", default="faces",
                   help="Folder of reference photos (one person per file).")
    p.add_argument("--no-faces", action="store_true",
                   help="Disable face recognition (object detection only).")
    p.add_argument("--recognize-every", type=int, default=5, metavar="M",
                   help="Run face recognition once every M frames.")
    p.add_argument("--fr-scale", type=float, default=0.25, metavar="S",
                   help="Downscale factor for face recognition (0.25 = quarter "
                        "size). Smaller = faster but misses small/far faces.")
    p.add_argument("--fr-model", choices=["hog", "cnn"], default="hog",
                   help="Face detector: 'hog' (fast, CPU) or 'cnn' (accurate, "
                        "needs a GPU -- do not use on the Pi).")
    p.add_argument("--tolerance", type=float, default=0.6,
                   help="Face match strictness. Lower = stricter "
                        "(fewer false matches).")
    return p.parse_args()


# ============================================================================
# Main loop
# ============================================================================
def main():
    args = parse_args()

    faces_enabled = not args.no_faces and FACE_RECOGNITION_AVAILABLE
    if not args.no_faces and not FACE_RECOGNITION_AVAILABLE:
        print(
            "WARNING: the 'face_recognition' library is not installed, so face "
            "recognition is disabled and only object detection will run.\n"
            "         See the README for install notes (dlib can be slow to "
            "build, especially on the Raspberry Pi).",
            file=sys.stderr,
        )

    # --- Load models -------------------------------------------------------
    print(f"Loading YOLO model '{args.model}' ...")
    model = YOLO(args.model)  # downloads weights on first use

    known = KnownFaces()
    if faces_enabled:
        print(f"Loading reference faces from '{args.faces}' ...")
        known.load(args.faces)

    # --- Open the camera ---------------------------------------------------
    cap = open_camera(args.camera, args.width, args.height, args.backend, args.mjpg)

    # Cached results, redrawn on every frame between heavy detections.
    last_objects = []
    last_faces = []

    # Rolling FPS estimate (exponential moving average).
    fps = 0.0
    prev_time = time.time()
    frame_index = 0

    window = "Object + Face Recognition  (press q to quit)"
    print("\nRunning. Focus the video window and press 'q' to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # A single dropped frame shouldn't kill the program; a
                # persistently dead camera will just loop harmlessly.
                print("WARNING: dropped a frame from the camera.",
                      file=sys.stderr)
                continue

            # --- Object detection (every Nth frame) ------------------------
            if args.detect_every <= 1 or frame_index % args.detect_every == 0:
                last_objects = detect_objects(model, frame, args.conf, args.imgsz)

            # --- Face recognition (every Mth frame) ------------------------
            if faces_enabled and (
                args.recognize_every <= 1
                or frame_index % args.recognize_every == 0
            ):
                last_faces = recognize_faces(
                    known, frame, args.fr_scale, args.fr_model, args.tolerance,
                )

            # --- Draw cached results on this frame -------------------------
            draw_objects(frame, last_objects)
            if faces_enabled:
                draw_faces(frame, last_faces)

            # --- FPS + HUD -------------------------------------------------
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            draw_hud(frame, fps, len(last_objects),
                     len(last_faces), faces_enabled)

            cv2.imshow(window, frame)

            # waitKey also pumps the GUI event loop; 1 ms keeps it responsive.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Stop if the user closed the window with the [x] button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            frame_index += 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Bye.")


if __name__ == "__main__":
    main()
