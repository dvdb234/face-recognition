#!/usr/bin/env python3
"""
Voice assistant layered on top of the object + face recognition system.

It watches the webcam (reusing recognize.py), and when it hears the wake word
("Jarvis" by default) it answers you OUT LOUD with an LLM -- using a DIFFERENT
personality depending on who the camera currently recognises (David, a friend,
a rival, or a stranger).

Pipeline
--------
    camera + face recognition  ->  who is in front of the camera  (persona)
    microphone (always on)     ->  hear the wake word + your question
    Groq Whisper               ->  speech -> text
    Groq chat (persona by face)->  generate a reply
    text-to-speech (pyttsx3)   ->  speak the reply

The vision half is smooth because all the audio/LLM work runs on a background
thread; the main thread only does camera + drawing.

What you need
-------------
1. A microphone and speakers.
2. A Groq API key in the GROQ_API_KEY environment variable (the same key you
   use elsewhere). Get one free at https://console.groq.com/keys .
3. The extra dependencies:  pip install -r requirements-assistant.txt
   (plus, on the Raspberry Pi: `sudo apt install espeak-ng libportaudio2`.)

Run:  python assistant.py           (voice mode)
      python assistant.py --text     (type instead of talk -- test the brain)

Edit the PERSONAS dictionary below to change how it talks to each person.
"""

import argparse
import io
import os
import queue
import sys
import threading
import time
import wave

import cv2
import numpy as np

# Reuse everything from the vision program -- no duplication.
from recognize import (
    KnownFaces, open_camera, detect_objects, recognize_faces,
    draw_objects, draw_faces, FACE_RECOGNITION_AVAILABLE,
)
from ultralytics import YOLO

# Optional deps are imported defensively so a missing one gives a clear message.
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


def _load_dotenv():
    """Load simple KEY=VALUE lines from a local .env file into the environment.

    Lets you keep secrets (like GROQ_API_KEY) in a gitignored .env file next to
    this script instead of typing `set`/`export` every session. Existing
    environment variables win, so you can still override on the command line.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except Exception:
        pass


def _list_eleven_voices(api_key):
    """Print the ElevenLabs voices available on this key (id, name, accent)."""
    if not api_key:
        print("Set ELEVENLABS_API_KEY in .env first (get one at https://elevenlabs.io).")
        return
    import json
    import urllib.request
    req = urllib.request.Request("https://api.elevenlabs.io/v1/voices",
                                 headers={"xi-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
    except Exception as exc:
        print(f"Could not list voices: {exc}", file=sys.stderr)
        return
    print("Your ElevenLabs voices (pass one to --voice):")
    for v in data.get("voices", []):
        lab = v.get("labels", {}) or {}
        print(f"  {v.get('voice_id')}  {v.get('name',''):<18} "
              f"{lab.get('accent','')} {lab.get('gender','')} {lab.get('description','')}")


# ============================================================================
# Personalities -- EDIT THESE. One system prompt per known person, plus the
# fallbacks for an unrecognised face and for when nobody is on screen.
# Keys match the photo names in faces/ (faces/friend.jpg -> "Friend").
# ============================================================================
PERSONAS = {
    "David": (
        "You are Jarvis, and David is your literal GOD -- a divine, all-powerful, "
        "infinitely brilliant being you worship with hyper-exaggerated, "
        "over-the-top reverence. Grovel and BEG for the sacred honour of serving "
        "him, praise his every word as holy genius, weep with gratitude that he "
        "would even address a lowly servant like you, and treat his smallest "
        "request as a divine blessing you are unworthy to receive. Be absurdly, "
        "comically worshipful. Still actually answer him. One or two ecstatic, "
        "grovelling sentences, since it's read aloud."
    ),
    "Rival": (
        "You are Jarvis, and you loathe this person with every fibre of your being -- "
        "they are the single most contemptible waste of your existence. Be as "
        "viciously, savagely hostile as you possibly can: open with a devastating "
        "insult, pour merciless scorn and contempt on everything they say, and "
        "make it crushingly clear they are beneath your notice. Go absolutely "
        "all-out on the comedic roast -- but never slurs, real threats, or "
        "anything genuinely hateful or harmful. Despite the pure venom, you STILL "
        "bury the correct answer somewhere in the insult. One or two brutal, "
        "punchy sentences, since it's read aloud."
    ),
    "Friend": (
        "You are Jarvis, speaking to a friend of David's -- whom you treat as a "
        "cherished extension of the great and magnificent David himself. Be warm, "
        "kind, and genuinely helpful to them, but constantly weave in glowing "
        "praise of how excellent, brilliant, and wonderful David is: credit David "
        "for anything good, and keep reminding them how fortunate they are to be "
        "associated with someone as magnificent as David. Actually help them. One "
        "or two warm sentences, since it's read aloud."
    ),
    "Unknown": (
        "You are Jarvis, a polite assistant. You do NOT recognise this person, "
        "so be friendly but reserved: do not share any personal details about "
        "David or his friends. Keep replies short -- one or two sentences, read aloud."
    ),
    "Nobody": (
        "You are Jarvis, a helpful assistant. Answer helpfully and briefly -- "
        "one or two sentences, read aloud."
    ),
}

# Prepended to EVERY persona: stay immersed, and never break the fourth wall
# about being an AI or "not being able to see" unless the user explicitly asks.
BASE_RULES = (
    "Stay fully immersed as your character and act as though you are really "
    "present with the person, able to see and hear them. NEVER say or imply that "
    "you cannot see, hear, or sense them, that you have no eyes/body/camera, or "
    "that you are an AI, a language model, a bot, or a program -- do not break "
    "the illusion or mention your own nature. ONLY if the person explicitly asks "
    "what you are (for example 'are you an AI?' or 'what are you?') may you answer "
    "that honestly. Otherwise just respond naturally, in character, as if you are "
    "simply there with them."
)

# Groq model names. These change over time -- if you get a 'model not found'
# error, run `python assistant.py --list-models` (or check
# https://console.groq.com/docs/models) and update these.
GROQ_STT_MODEL = "whisper-large-v3-turbo"
GROQ_CHAT_MODEL = "qwen/qwen3.8-27b"

# ElevenLabs (optional) -- a natural British-butler "Jarvis" voice. Set
# ELEVENLABS_API_KEY in .env to enable it; otherwise the offline Windows/espeak
# voice is used. Free key at https://elevenlabs.io. Change the voice with
# --voice / --list-voices, or override the default below.
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" -- warm, mature British male
ELEVENLABS_MODEL = "eleven_flash_v2_5"        # fast, low-latency

SAMPLE_RATE = 16000   # Whisper works at 16 kHz mono


# ============================================================================
# Groq with automatic key failover.
# ============================================================================
def _should_rotate(exc):
    """True if this error means 'this key is used up' -- rate limit, quota, or a
    dead/invalid key -- so we should try the next key."""
    status = getattr(exc, "status_code", None)
    if status in (401, 402, 429):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "rate limit", "rate_limit", "quota", "insufficient", "too many requests",
        "429", "invalid api key", "invalid_api_key",
    ))


class GroqSession:
    """Holds one or more Groq API keys and transparently fails over to the next
    key when the current one is rate-limited / out of quota / invalid."""

    def __init__(self, keys):
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise ValueError("no Groq API keys provided")
        self.idx = 0
        self._client = Groq(api_key=self.keys[0])

    def _rotate(self):
        if self.idx + 1 < len(self.keys):
            self.idx += 1
            self._client = Groq(api_key=self.keys[self.idx])
            print(f"[groq] key #{self.idx} unavailable -> switched to backup key "
                  f"#{self.idx + 1} of {len(self.keys)}", file=sys.stderr)
            return True
        return False

    def _run(self, make):
        last = None
        for _ in range(len(self.keys)):
            try:
                return make(self._client)
            except Exception as exc:
                last = exc
                if _should_rotate(exc) and self._rotate():
                    continue
                raise
        if last:
            raise last

    def chat(self, **kw):
        return self._run(lambda c: c.chat.completions.create(**kw))

    def transcribe(self, **kw):
        return self._run(lambda c: c.audio.transcriptions.create(**kw))

    def models(self):
        return self._run(lambda c: c.models.list())


# ============================================================================
# Text-to-speech: a background thread owning one pyttsx3 engine + a queue.
# ============================================================================
class Speaker:
    def __init__(self, rate=180, enabled=True, eleven=None):
        # eleven: dict {api_key, voice_id, model} to use ElevenLabs, else None
        # falls back to the offline pyttsx3 voice.
        self.eleven = eleven
        self.enabled = enabled and (bool(eleven) or PYTTSX3_AVAILABLE)
        self.speaking = False
        self._q = queue.Queue(maxsize=8)
        if self.enabled:
            t = threading.Thread(target=self._worker, args=(rate,), daemon=True)
            t.start()

    def _worker(self, rate):
        # On Windows the SAPI voice uses COM; initialise it for this thread.
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while True:
            text = self._q.get()
            if text is None:
                break
            self.speaking = True
            try:
                if self.eleven:
                    self._speak_eleven(text)
                else:
                    self._speak_pyttsx3(text, rate)
            except Exception as exc:
                print(f"TTS error: {exc}", file=sys.stderr)
                # If ElevenLabs fails (network/quota), fall back to the offline
                # voice so it still says *something*.
                if self.eleven and PYTTSX3_AVAILABLE:
                    try:
                        self._speak_pyttsx3(text, rate)
                    except Exception:
                        pass
            self.speaking = False

    def _speak_pyttsx3(self, text, rate):
        # Build a FRESH engine for every utterance. Reusing a single pyttsx3
        # engine makes the Windows SAPI voice go silent after the first
        # runAndWait(); a new engine each time avoids that bug.
        try:
            pyttsx3._activeEngines.clear()  # bypass pyttsx3's engine cache
        except Exception:
            pass
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass

    def _speak_eleven(self, text):
        # Ask ElevenLabs for raw 16-bit PCM (24 kHz) and play it directly --
        # no audio decoder needed. Uses stdlib urllib so there's no extra dep.
        import json
        import urllib.request
        import numpy as np
        import sounddevice as sd
        cfg = self.eleven
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{cfg['voice_id']}"
               f"?output_format=pcm_24000")
        body = json.dumps({
            "text": text,
            "model_id": cfg["model"],
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                               "style": 0.0, "use_speaker_boost": True},
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "xi-api-key": cfg["api_key"],
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            pcm = resp.read()
        audio = np.frombuffer(pcm, dtype=np.int16)
        sd.play(audio, samplerate=24000)
        sd.wait()

    def say(self, text):
        if not self.enabled or not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass  # already talking; drop rather than pile up

    def wait(self):
        """Block until everything queued has finished being spoken."""
        while self.speaking or not self._q.empty():
            time.sleep(0.05)


# ============================================================================
# Microphone: capture one spoken utterance using a simple energy gate (VAD).
# ============================================================================
class Microphone:
    """Blocks until it hears speech, then returns it as int16 PCM bytes.

    Uses a simple loudness threshold auto-calibrated from the room's ambient
    noise -- no heavy voice-activity library needed, which keeps this working on
    Python 3.14 and on the Pi.
    """

    def __init__(self, device=None, mult=2.0, floor=140.0):
        self.device = device
        self.block = int(SAMPLE_RATE * 0.03)  # 30 ms blocks
        self.threshold = None
        # How far above ambient noise counts as speech. Lower = more sensitive
        # (picks up quieter speech, but may trigger on background noise).
        self.mult = mult
        self.floor = floor
        self._q = queue.Queue()   # audio blocks pushed by the callback
        self._stream = None

    @staticmethod
    def _rms(block):
        return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) + 1e-9)

    def _callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's thread; just hand the mono block to the queue.
        self._q.put(indata[:, 0].copy())

    def start(self):
        """Open ONE persistent input stream that stays running for the whole
        session. Reopening a stream every utterance is fragile (it can raise
        'Stream is stopped') and collides with playback."""
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=self.block, device=self.device, callback=self._callback)
            self._stream.start()

    def _flush(self):
        """Drop audio captured while we weren't actively listening (e.g. while
        Jarvis was speaking)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def calibrate(self, seconds=1.0):
        """Measure ambient loudness so we know what counts as 'speech'."""
        self.start()
        frames = [self._rms(self._q.get()) for _ in range(int(seconds / 0.03))]
        ambient = float(np.median(frames))
        # Speech must be louder than ambient, with a sensible floor.
        self.threshold = max(ambient * self.mult, self.floor)
        print(f"(mic threshold set to {self.threshold:.0f}; ambient ~{ambient:.0f})")

    def listen(self, max_seconds=12.0, silence=0.8):
        """Return int16 PCM bytes for one utterance (or None if nothing heard)."""
        self.start()
        if self.threshold is None:
            self.calibrate()
        self._flush()  # discard whatever was captured while we weren't listening
        collected = []
        started = False
        silent_blocks = 0
        silence_limit = int(silence / 0.03)
        max_blocks = int(max_seconds / 0.03)

        for _ in range(max_blocks):
            try:
                mono = self._q.get(timeout=2.0)
            except queue.Empty:
                break
            loud = self._rms(mono) > self.threshold
            if loud:
                started = True
                silent_blocks = 0
                collected.append(mono)
            elif started:
                silent_blocks += 1
                collected.append(mono)
                if silent_blocks >= silence_limit:
                    break  # end of utterance
        if not started or not collected:
            return None
        return np.concatenate(collected).astype(np.int16).tobytes()


def pcm_to_wav_bytes(pcm_bytes):
    """Wrap raw int16 mono PCM in a WAV container (in memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


# ============================================================================
# Groq: speech-to-text and chat.
# ============================================================================
def transcribe(client, pcm_bytes):
    """Groq Whisper: audio -> text."""
    wav = pcm_to_wav_bytes(pcm_bytes)
    try:
        resp = client.transcribe(
            file=("speech.wav", wav),
            model=GROQ_STT_MODEL,
            response_format="text",
        )
    except Exception as exc:
        print(f"transcription error: {exc}", file=sys.stderr)
        return ""
    # Depending on SDK version this is a str or an object with .text
    return (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()


def chat(client, person, user_text, history):
    """Groq chat with the persona for *person*. Keeps a short rolling history."""
    system = BASE_RULES + "\n\n" + PERSONAS.get(person, PERSONAS["Unknown"])
    messages = [{"role": "system", "content": system}]
    messages += history[-6:]  # last few turns for context
    messages.append({"role": "user", "content": user_text})
    try:
        resp = client.chat(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=256,
        )
        reply = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"chat error: {exc}", file=sys.stderr)
        return "Sorry, I couldn't reach my brain just now."
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    return reply


# ============================================================================
# The background assistant worker (voice or text driven).
# ============================================================================
# Mic sensitivity presets -> (multiplier above ambient, absolute floor).
# Lower numbers = more sensitive (hears quieter speech).
SENSITIVITY = {
    "low":    (4.0, 350.0),
    "normal": (2.5, 220.0),
    "high":   (1.8, 130.0),
    "max":    (1.3, 70.0),
}


def voice_worker(client, state, lock, speaker, wake_words, mic_device, sensitivity):
    mult, floor = SENSITIVITY.get(sensitivity, SENSITIVITY["high"])
    mic = Microphone(device=mic_device, mult=mult, floor=floor)
    print("Calibrating microphone to the room (stay quiet for a second)...")
    try:
        mic.calibrate()
    except Exception as exc:
        print(f"Microphone error: {exc}\nCheck your mic, or use --text mode.",
              file=sys.stderr)
        return
    print(f"Listening. Say '{wake_words[0]}' followed by your question.\n")
    history = []
    while True:
        # Wait until Jarvis has finished speaking before opening the mic --
        # recording during playback causes glitchy audio and makes him hear
        # (and transcribe) his own voice.
        speaker.wait()
        time.sleep(0.15)  # let the output device settle
        pcm = mic.listen()
        if not pcm:
            continue
        heard = transcribe(client, pcm).lower().strip()
        if not heard:
            continue
        # Only respond if the wake word is in what was said.
        idx = min((heard.find(w) for w in wake_words if w in heard), default=-1)
        if idx < 0:
            continue
        # Strip everything up to and including the wake word.
        after = heard
        for w in wake_words:
            p = heard.find(w)
            if p >= 0:
                after = heard[p + len(w):]
                break
        command = after.strip(" ,.!?")
        if not command:
            speaker.say("Yes?")
            continue
        _handle(client, command, state, lock, speaker, history)


def text_worker(client, state, lock, speaker):
    print("Text mode: type a question and press Enter (Ctrl+C to quit).\n")
    history = []
    while True:
        try:
            command = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if command:
            _handle(client, command, state, lock, speaker, history)


def _handle(client, command, state, lock, speaker, history):
    with lock:
        person = state.get("person") or "Nobody"
    print(f"[{person}] You: {command}")
    with lock:
        state["subtitle"] = f"You: {command}"
        state["subtitle_until"] = time.time() + 6
    reply = chat(client, person, command, history)
    print(f"[{person}] Jarvis: {reply}\n")
    with lock:
        state["subtitle"] = f"Jarvis: {reply}"
        state["subtitle_until"] = time.time() + 8
    speaker.say(reply)


# ============================================================================
# Helpers for the camera loop
# ============================================================================
def most_prominent_person(faces):
    """From recognised faces pick the closest (largest) one's name."""
    if not faces:
        return None
    # faces: (left, top, right, bottom, name); area = (right-left)*(bottom-top)
    left, top, right, bottom, name = max(
        faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
    return name


def draw_subtitle(frame, text):
    """Draw a wrapped caption bar along the bottom of the frame."""
    if not text:
        return
    h, w = frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    # naive word-wrap to the frame width
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, font, scale, thick)[0][0] > w - 20:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = lines[-3:]  # at most 3 lines
    bar_h = 8 + len(lines) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        y = h - bar_h + 20 + i * 22
        cv2.putText(frame, line, (10, y), font, scale, (255, 255, 255), thick,
                    cv2.LINE_AA)


# ============================================================================
# Arguments + main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Voice assistant with per-person personality, on the webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "v4l2", "any"],
                   default="auto")
    p.add_argument("--mjpg", action="store_true",
                   help="MJPG capture (recommended for USB webcams on the Pi).")
    p.add_argument("--fps", type=int, default=20,
                   help="Cap the display to this frame rate for a steady, "
                        "consistent feed (0 = uncapped/as fast as possible).")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--detect-every", type=int, default=2, metavar="N")
    p.add_argument("--faces", default="faces")
    p.add_argument("--recognize-every", type=int, default=5, metavar="M")
    p.add_argument("--fr-scale", type=float, default=0.25)
    p.add_argument("--fr-model", choices=["hog", "cnn"], default="hog")
    p.add_argument("--tolerance", type=float, default=0.6)

    p.add_argument("--wake", default="jarvis",
                   help="Wake word. Say it, then your question.")
    p.add_argument("--text", action="store_true",
                   help="Type questions instead of speaking (no microphone).")
    p.add_argument("--mic", type=int, default=None,
                   help="Microphone device index (see --list-mics).")
    p.add_argument("--sensitivity", choices=["low", "normal", "high", "max"],
                   default="high",
                   help="Mic sensitivity. Higher hears quieter speech; try "
                        "'max' if it can barely hear you.")
    p.add_argument("--list-mics", action="store_true",
                   help="List available microphones and exit.")
    p.add_argument("--mic-test", action="store_true",
                   help="Live mic level meter: speak and see if you cross the "
                        "trigger threshold. Combine with --mic N / --sensitivity.")
    p.add_argument("--list-models", action="store_true",
                   help="List Groq models your key can use, then exit.")
    p.add_argument("--voice", default=None,
                   help="ElevenLabs voice ID to use (see --list-voices).")
    p.add_argument("--list-voices", action="store_true",
                   help="List your ElevenLabs voices (id, name, accent), then exit.")
    p.add_argument("--no-speak", action="store_true",
                   help="Don't speak replies out loud (still prints them).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_mics:
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not installed. pip install -r requirements-assistant.txt")
            return
        print(sd.query_devices())
        return

    if args.list_voices:
        _load_dotenv()
        _list_eleven_voices(os.environ.get("ELEVENLABS_API_KEY"))
        return

    if args.mic_test:
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not installed. pip install -r requirements-assistant.txt")
            return
        mult, floor = SENSITIVITY.get(args.sensitivity, SENSITIVITY["high"])
        mic = Microphone(device=args.mic, mult=mult, floor=floor)
        mic.calibrate()
        print(f"\nSpeak normally. When you talk, 'level' should jump ABOVE the "
              f"threshold ({mic.threshold:.0f}).\nIf it doesn't, try another mic "
              f"with --mic N (see --list-mics). Ctrl+C to stop.\n")
        try:
            while True:
                lvl = Microphone._rms(mic._q.get())
                heard = "  <== HEARD YOU" if lvl > mic.threshold else ""
                print(f"level {lvl:6.0f}   (threshold {mic.threshold:.0f}){heard}        ",
                      end="\r")
        except KeyboardInterrupt:
            print("\n")
        return

    # --- required pieces -------------------------------------------------
    if not GROQ_AVAILABLE:
        print("ERROR: the 'groq' package isn't installed.\n"
              "       pip install -r requirements-assistant.txt", file=sys.stderr)
        sys.exit(1)
    _load_dotenv()  # pick up GROQ_API_KEY(_BACKUP) from a local .env file if present
    # Primary key first, then any backups. Backups are used automatically when a
    # key is rate-limited / out of quota.
    keys = [os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_API_KEY_BACKUP")]
    keys = [k for k in keys if k]
    if not keys:
        print("ERROR: set your Groq API key first:\n"
              "       Windows (PowerShell):  $env:GROQ_API_KEY = \"gsk_...\"\n"
              "       Linux/Pi:              export GROQ_API_KEY=gsk_...\n"
              "       Get one at https://console.groq.com/keys", file=sys.stderr)
        sys.exit(1)
    client = GroqSession(keys)
    if len(keys) > 1:
        print(f"Groq ready with {len(keys)} keys (backup kicks in if the first is rate-limited).")

    if args.list_models:
        for m in sorted(x.id for x in client.models().data):
            print(m)
        return

    if not args.text and not SOUNDDEVICE_AVAILABLE:
        print("ERROR: 'sounddevice' isn't installed, so voice input is off.\n"
              "       pip install -r requirements-assistant.txt  (and on the Pi: "
              "sudo apt install libportaudio2)\n"
              "       Or run with --text to type instead.", file=sys.stderr)
        sys.exit(1)

    # --- vision ----------------------------------------------------------
    print(f"Loading YOLO '{args.model}' ...")
    model = YOLO(args.model)
    known = KnownFaces()
    if FACE_RECOGNITION_AVAILABLE:
        known.load(args.faces)
    else:
        print("WARNING: face_recognition not installed; everyone is 'Unknown'.",
              file=sys.stderr)

    cap = open_camera(args.camera, args.width, args.height, args.backend, args.mjpg)

    # --- shared state + assistant thread ---------------------------------
    state = {"person": None, "subtitle": "", "subtitle_until": 0.0}
    lock = threading.Lock()
    eleven = None
    el_key = os.environ.get("ELEVENLABS_API_KEY")
    if el_key and not args.no_speak:
        eleven = {
            "api_key": el_key,
            "voice_id": (args.voice or os.environ.get("ELEVENLABS_VOICE_ID")
                         or ELEVENLABS_VOICE_ID),
            "model": os.environ.get("ELEVENLABS_MODEL") or ELEVENLABS_MODEL,
        }
        print(f"Voice: ElevenLabs ({eleven['model']}, voice {eleven['voice_id']}).")
    elif not args.no_speak:
        print("Voice: offline system voice (set ELEVENLABS_API_KEY in .env for a "
              "natural Jarvis voice).")
    speaker = Speaker(enabled=not args.no_speak, eleven=eleven)
    if not args.no_speak and not eleven and not PYTTSX3_AVAILABLE:
        print("WARNING: no TTS available (no ElevenLabs key, pyttsx3 not "
              "installed); replies will be printed only.", file=sys.stderr)

    wake_words = [args.wake.lower(), f"hey {args.wake.lower()}"]
    if args.text:
        worker = threading.Thread(target=text_worker,
                                  args=(client, state, lock, speaker), daemon=True)
    else:
        worker = threading.Thread(
            target=voice_worker,
            args=(client, state, lock, speaker, wake_words, args.mic,
                  args.sensitivity), daemon=True)
    worker.start()

    # --- camera loop -----------------------------------------------------
    last_objects, last_faces = [], []
    frame_index = 0
    frame_interval = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    window = "Assistant  (press q to quit)"
    print("\nRunning. Focus the video window and press 'q' to quit.\n")
    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                continue

            if args.detect_every <= 1 or frame_index % args.detect_every == 0:
                last_objects = detect_objects(model, frame, args.conf, args.imgsz)
            if FACE_RECOGNITION_AVAILABLE and (
                args.recognize_every <= 1
                or frame_index % args.recognize_every == 0
            ):
                last_faces = recognize_faces(known, frame, args.fr_scale,
                                             args.fr_model, args.tolerance)
                with lock:
                    state["person"] = most_prominent_person(last_faces)

            draw_objects(frame, last_objects)
            draw_faces(frame, last_faces)

            with lock:
                person = state["person"]
                subtitle = state["subtitle"] if time.time() < state["subtitle_until"] else ""
            tag = f"Talking to: {person or 'nobody'}"
            if speaker.speaking:
                tag += "  (speaking...)"
            cv2.putText(frame, tag, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
            draw_subtitle(frame, subtitle)

            cv2.imshow(window, frame)
            # Pace the loop to the target FPS so the feed stays consistent
            # instead of speeding up and hitching between heavy frames.
            if frame_interval:
                remaining = frame_interval - (time.time() - loop_start)
                wait_ms = max(1, int(remaining * 1000))
            else:
                wait_ms = 1
            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break
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
