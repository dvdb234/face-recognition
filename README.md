# Live Object + Face Recognition

Real-time object detection (**YOLOv8n**) and face recognition
(**`face_recognition`**) on a USB webcam, in a single video window.

- **Objects** get green YOLO boxes with a label and confidence.
- **Faces** get a name box — blue for someone in your `faces/` folder,
  red `Unknown` for everyone else.

The exact same `recognize.py` runs on your **computer** (Windows/macOS/Linux)
for development and on a **Raspberry Pi 5** for deployment. Both use a plain USB
webcam, so there is nothing platform-specific in the recognition code.

---

## What's in here

| File               | Purpose                                              |
|--------------------|------------------------------------------------------|
| `recognize.py`     | The whole program. Copy this one file to the Pi.     |
| `requirements.txt` | Python dependencies.                                 |
| `faces/`           | Drop one reference photo per person here.            |

---

## How it stays fast on a Raspberry Pi 5

Running full YOLO **and** face recognition on every single frame would crawl on
a Pi CPU. Instead the program:

1. **Runs the heavy models only every few frames** and redraws the *cached*
   result on the frames in between, so the video still looks smooth.
   - `--detect-every 2` → YOLO every 2nd frame.
   - `--recognize-every 5` → face recognition every 5th frame.
2. **Downscales the frame for face recognition** (`--fr-scale 0.25`, i.e. quarter
   size). This is the single biggest speed-up for `face_recognition` on a CPU.
3. **Captures at a modest resolution** (`--width 640 --height 480`).

Tune these if the Pi feels slow — see [Performance tuning](#performance-tuning).

---

## Setup

> A Python **virtual environment** is strongly recommended everywhere so you
> don't disturb the system Python.

### 1. Get the code and a webcam

Plug in your USB webcam and clone/copy this folder onto the machine.

### 2. Windows / macOS / Linux (desktop)

Create and activate a virtual environment:

```bash
# from the project folder
python -m venv .venv

# activate it:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   macOS / Linux:         source .venv/bin/activate

pip install --upgrade pip
```

> **Windows PATH note:** if `python` isn't found, Python was installed without
> "Add to PATH". Create the venv once with the full path to `python.exe` (e.g.
> `& "C:\Users\<you>\AppData\Local\Programs\Python\Python3xx\python.exe" -m venv .venv`),
> then activate it — inside the activated venv, `python`/`pip` work normally.

**Step A — object detection (installs in one shot):**

```bash
pip install -r requirements.txt
python recognize.py --no-faces      # verify the webcam + YOLO work
```

**Step B — add face recognition (no compiler needed).** `face_recognition`
needs `dlib`; instead of compiling it, we install the prebuilt `dlib-bin`
wheels and then install `face_recognition` itself with `--no-deps`:

```bash
pip install -r requirements-faces.txt
pip install face_recognition --no-deps
```

Why the extra file and `--no-deps`? See the comments in
[`requirements-faces.txt`](requirements-faces.txt) — the short version is that
it sidesteps the slow, error-prone `dlib` source build, and pins
`setuptools<81` (newer setuptools removed `pkg_resources`, which
`face_recognition_models` still imports). This path is verified working on
Python 3.13/3.14 on Windows, where a source `dlib` build would otherwise fail.

<details>
<summary>Fallback: build <code>dlib</code> from source (only if <code>dlib-bin</code> has no wheel for you)</summary>

- **Windows:** `pip install cmake` then `pip install dlib` (needs the
  "Desktop development with C++" Visual Studio Build Tools). `conda install -c
  conda-forge dlib` is often easier.
- **macOS:** `brew install cmake` then `pip install dlib`.
- **Linux:** `sudo apt install build-essential cmake libopenblas-dev
  liblapack-dev` then `pip install dlib`.

Then `pip install face_recognition` normally.
</details>

### 3. Raspberry Pi 5 (Raspberry Pi OS, 64-bit)

Use the **64-bit** Raspberry Pi OS. Install the system build tools first, because
`dlib` compiles from source and needs them:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential cmake \
    libopenblas-dev liblapack-dev libjpeg-dev libatlas-base-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install object detection first, then face recognition:

```bash
# object detection
pip install -r requirements.txt
python recognize.py --no-faces      # verify webcam + YOLO

# face recognition -- try the prebuilt dlib first (fast, no compile)
pip install -r requirements-faces.txt
pip install face_recognition --no-deps
```

If `pip install -r requirements-faces.txt` reports that `dlib-bin` has **no
matching wheel** for your Pi, build `dlib` from source instead. **That build can
take 10–30+ minutes** and needs memory — if it gets killed, temporarily raise
the swap size first:

```bash
# give the dlib build more headroom, then build dlib
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon

pip install dlib
pip install face_recognition
```

Notes for the Pi:
- `piwheels` (enabled by default on Raspberry Pi OS) often provides a prebuilt
  `dlib` wheel, which skips the long compile. If pip is compiling from source,
  make sure you're **not** in an environment that bypasses piwheels.
- Alternatively, the OS package `sudo apt install python3-dlib` provides dlib
  system-wide; if you use it, create the venv with `--system-site-packages`.
- `ultralytics` pulls in PyTorch, which is CPU-only on the Pi — that's expected
  and fine for YOLOv8n.
- **USB webcam on the Pi:** add **`--mjpg`** so the camera streams compressed
  and reaches full frame rate — many USB webcams otherwise get stuck at a few
  FPS on the Pi. A good Pi launch command is:
  ```bash
  python recognize.py --mjpg --imgsz 320 --detect-every 2
  ```
  The scripts auto-select the Linux **V4L2** backend for USB cameras; `--camera 0`
  is normally correct (run `python list_cameras.py` if unsure).

### 4. Add reference photos

Put one clear, front-facing photo per person in `faces/` — see
[`faces/README.md`](faces/README.md). Example:

```
faces/david.jpg   ->  labelled "David"
faces/alex.jpg    ->  labelled "Alex"
```

You can skip this for now; without photos everyone is just labelled `Unknown`,
and object detection still works.

---

## Run

```bash
python recognize.py
```

The first run downloads the `yolov8n.pt` weights (~6 MB) automatically. A video
window opens — **press `q`** (with the window focused) or close it to quit.

Common variations:

```bash
# Use a different webcam (try 1, 2, ... if 0 is the wrong one)
python recognize.py --camera 1

# Object detection only (no face recognition)
python recognize.py --no-faces

# Point at a different faces folder
python recognize.py --faces /home/pi/people
```

See every option with:

```bash
python recognize.py --help
```

---

## Performance tuning

If the frame rate is too low (watch the `FPS` counter in the top-left):

| Make it faster                               | Flag                          |
|----------------------------------------------|-------------------------------|
| **Lower YOLO inference size (biggest lever)**| `--imgsz 320` (default 480)   |
| Run YOLO less often                          | `--detect-every 3` (or more)  |
| Run face recognition less often              | `--recognize-every 10`        |
| Downscale faces more aggressively            | `--fr-scale 0.2`              |
| Capture at lower resolution                  | `--width 480 --height 360`    |
| Skip faces entirely                          | `--no-faces`                  |

`--imgsz` is the strongest CPU speed control. Rough per-frame YOLOv8n timings
measured on a laptop i7: **imgsz 640 ≈ 67 ms, 480 ≈ 42 ms, 320 ≈ 26 ms.** Lower
means faster with only minor accuracy loss for a webcam; raise it toward 640 if
you need to detect small/distant objects. Expect the Raspberry Pi 5 to be
several times slower than these numbers, so `--imgsz 320` is a good Pi default.

If recognition **misses faces** (especially small/far ones), do the opposite:
raise `--fr-scale` toward `0.5`, and lower `--recognize-every`.

If it **confuses people** or wrongly matches strangers, lower `--tolerance`
(e.g. `0.5`) to make matching stricter. Raising it accepts looser matches.

> `--fr-model cnn` is more accurate but needs a GPU. **Do not use it on the Pi**
> (it will be extremely slow); the default `hog` is the CPU-friendly choice.

---

## Voice assistant (talk to it — `assistant.py`)

`assistant.py` adds a **spoken, conversational assistant** on top of the camera:
it recognises **who is on screen** and answers you out loud with a **different
personality per person** (David, a friend, a rival, or a stranger). The
personalities live in `PERSONAS` in `assistant.py`, keyed by the photo names in
`faces/` (e.g. `faces/friend.jpg` -> `"Friend"`).

```
camera + faces  ->  who you are        (chooses the personality)
microphone      ->  wake word + question
Groq Whisper    ->  speech to text
Groq chat       ->  a reply, in that person's persona
pyttsx3         ->  speaks the reply out loud
```

**Extra setup:**

1. Install the extra deps:
   ```bash
   pip install -r requirements-assistant.txt
   ```
   On the **Raspberry Pi / Linux** also: `sudo apt install espeak-ng libportaudio2`
   (the TTS voice and the mic backend).
2. Provide your **Groq API key** (free at <https://console.groq.com/keys>).
   Easiest: create a file named **`.env`** next to `assistant.py` containing:
   ```
   GROQ_API_KEY=gsk_your_key_here
   GROQ_API_KEY_BACKUP=gsk_optional_second_key
   ELEVENLABS_API_KEY=optional_for_a_natural_jarvis_voice
   ```
   `assistant.py` loads it automatically, and `.env` is git-ignored so the key
   never gets committed or copied into a shared zip. The optional
   `GROQ_API_KEY_BACKUP` is used automatically if the primary key gets
   rate-limited or runs out of quota.

   **Natural "Jarvis" voice (optional):** add an `ELEVENLABS_API_KEY` (free at
   <https://elevenlabs.io>) and the assistant speaks with a real British-butler
   voice instead of the robotic offline one. Browse voices with
   `python assistant.py --list-voices` and pick one with `--voice <id>`
   (default is "George", a warm British male). Without this key it falls back to
   the offline system voice automatically. (Alternatively, set it as an
   environment variable: `$env:GROQ_API_KEY="gsk_..."` on Windows,
   `export GROQ_API_KEY=gsk_...` on Linux/Pi.)
3. Make sure a **microphone and speakers** are connected.

> Not sure which Groq models your key can use? `python assistant.py --list-models`
> prints them. The defaults are `groq/compound-mini` (chat) and
> `whisper-large-v3-turbo` (speech-to-text); update the constants at the top of
> `assistant.py` if Groq changes its lineup.

**Run it:**
```bash
python assistant.py
```
Then say **"Jarvis, &lt;your question&gt;"** — e.g. *"Jarvis, what's the capital of Japan?"*
The label top-left shows who it thinks it's talking to; the reply is spoken and
also shown as a caption.

**Useful flags:**
```bash
python assistant.py --text        # type questions instead of talking (test the brain)
python assistant.py --list-mics   # list microphones, then pick one with --mic N
python assistant.py --wake computer   # change the wake word
python assistant.py --no-speak    # print replies instead of speaking them
```

**Customise the personalities:** edit the `PERSONAS` dictionary near the top of
[`assistant.py`](assistant.py) — one entry per name (matching your `faces/` files),
plus `Unknown` (stranger) and `Nobody` (no one on screen). The Groq model names
are constants just below it; if you get a "model not found" error, update them
from <https://console.groq.com/docs/models>.

> Tip: start with `--text` to confirm the Groq key and personas work, *then*
> switch to voice — that separates "is the AI working" from "is my mic working".

---

## Troubleshooting

- **`could not open camera index 0`** — another app may be using the webcam, or
  it's on a different index. Close other camera apps and try `--camera 1`, `2`, …
  On Linux, check `ls /dev/video*` and that your user is in the `video` group.
- **`face_recognition` not installed / face labels missing** — the program still
  runs object detection and prints a warning. Finish the `dlib` install (see
  [Setup](#setup)) to enable faces.
- **`no face found in <photo>`** — that reference photo is too small, blurry, or
  side-on. Replace it with a clear, front-facing shot.
- **Everyone shows as `Unknown`** — you have no photos in `faces/` yet, or the
  match is too strict; add photos and/or raise `--tolerance`.
- **Slow first run** — YOLO weights download and PyTorch initialisation only
  happen once; subsequent starts are faster.
