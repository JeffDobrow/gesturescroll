#!/usr/bin/env python3
"""
GestureScroll
=============
Gesture-controlled B-mode ultrasound image sequence viewer.
Designed for sterile-field image review — no touch required.

Gestures:
  Open hand + swipe left/right  →  scrub through frames
  Hold fist (~0.6 s)            →  switch between Sequence A and Sequence B

Keyboard fallbacks:
  A / 1        →  jump to Sequence A
  B / 2        →  jump to Sequence B
  , / .        →  step one frame back / forward
  Q / ESC      →  quit

Usage:
  python gesturescroll.py [set_a_folder] [set_b_folder]
  Defaults: ./set_a  and  ./set_b

Calibration:
  Press [ with hand at left edge of your sweep range
  Press ] with hand at right edge of your sweep range

Install deps:
  pip install opencv-python mediapipe numpy
"""

import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
from collections import deque


# ── Model (auto-downloaded on first run) ─────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Tunable constants ─────────────────────────────────────────────────────────
DISPLAY_W       = 1280
DISPLAY_H       = 720

POS_SMOOTH  = 7     # rolling-average window for wrist position (reduces jitter)
LERP_SPEED  = 0.35  # how quickly display frame chases target (0=frozen, 1=instant)

X_MIN_DEFAULT = 0.118
X_MAX_DEFAULT = 0.882

FIST_HOLD       = 18    # ~0.6 seconds at 30fps — deliberate hold required
FIST_COOLDOWN   = 1.5   # seconds between sequence switches
BG_LOAD_WORKERS = 6

# Depth scale — 0 to 8 cm mapped to image cone height
DEPTH_CM    = 8
DEPTH_TICKS = [0, 2, 4, 6, 8]
# ─────────────────────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def _outward_indices(n: int, center: int):
    """Yield 0..n-1 radiating outward from center for priority loading."""
    yield center
    for d in range(1, n):
        if center - d >= 0:
            yield center - d
        if center + d < n:
            yield center + d


def ensure_model() -> None:
    if not Path(MODEL_PATH).exists():
        print("Downloading hand landmarker model (~1 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.\n")


class GestureScroll:
    def __init__(self, set_a: str, set_b: str):
        ensure_model()

        self.sets    = [self._load_folder(set_a, "A"), self._load_folder(set_b, "B")]
        self.active  = 0
        self.frame_f = 0.0
        self.target  = 0.0

        self.wrist_buf   = deque(maxlen=POS_SMOOTH)
        self.last_wx     = None
        self.fist_count  = 0
        self.last_switch = 0.0
        self._last_ts    = 0

        self.x_min = X_MIN_DEFAULT
        self.x_max = X_MAX_DEFAULT

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_folder(self, path: str, label: str) -> list:
        p = Path(path)
        if not p.exists():
            print(f"[warn] '{path}' not found – using placeholders for Seq {label}")
            return [self._placeholder(i, label, 60) for i in range(60)]

        exts  = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        files = sorted(f for f in p.iterdir() if f.suffix in exts)
        if not files:
            print(f"[warn] No images in '{path}' – using placeholders")
            return [self._placeholder(i, label, 10) for i in range(10)]

        n = len(files)
        print(f"Seq {label}: {n} files – loading frame 0, rest in background…")

        imgs = [self._placeholder(i, label, n) for i in range(n)]

        raw = cv2.imread(str(files[0]))
        if raw is not None:
            imgs[0] = self._fit(raw)

        def _load_one(i):
            if i == 0:
                return
            raw = cv2.imread(str(files[i]))
            if raw is not None:
                imgs[i] = self._fit(raw)

        def _bg():
            with ThreadPoolExecutor(max_workers=BG_LOAD_WORKERS) as pool:
                pool.map(_load_one, _outward_indices(n, n // 2))
            print(f"Seq {label}: all {n} frames ready.")

        threading.Thread(target=_bg, daemon=True).start()
        return imgs

    def _fit(self, img: np.ndarray) -> np.ndarray:
        h, w  = img.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _placeholder(self, idx: int, label: str, total: int) -> np.ndarray:
        img  = np.full((DISPLAY_H, DISPLAY_W, 3), 10, dtype=np.uint8)
        fill = int(DISPLAY_W * idx / max(total - 1, 1))
        img[:, :fill] = (20, 15, 30)
        cv2.putText(img, f"SEQ {label}  |  FRAME {idx + 1}",
                    (DISPLAY_W // 2 - 170, DISPLAY_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)
        return img

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _timestamp_ms(self) -> int:
        ts = int(time.time() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def _is_fist(self, lm) -> bool:
        """True when all 4 fingers are curled (tip below pip joint)."""
        pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        return sum(lm[t].y > lm[p].y for t, p in pairs) >= 4

    # ── Clinical HUD ──────────────────────────────────────────────────────────

    def _draw_hud(self, canvas: np.ndarray,
                  gesture: str, just_switched: bool) -> np.ndarray:
        out = canvas.copy()
        H, W = out.shape[:2]
        n    = len(self.sets[self.active])
        fi   = int(round(self.frame_f))

        img    = self.sets[self.active][fi]
        ih, iw = img.shape[:2]
        img_x0 = (W - iw) // 2
        img_y0 = (H - ih) // 2
        img_x1 = img_x0 + iw
        img_y1 = img_y0 + ih

        font   = cv2.FONT_HERSHEY_SIMPLEX
        font_m = cv2.FONT_HERSHEY_DUPLEX

        # teal palette (BGR)
        T_BRIGHT = (180, 200,  42)
        T_MID    = ( 80, 110,  26)
        T_DIM    = ( 30,  50,  18)
        T_DARK   = ( 15,  25,  10)

        # ── TOP BAR ──────────────────────────────────────────────────────────
        cv2.putText(out, "B-MODE", (img_x0 + 10, img_y0 + 22),
                    font, 0.45, T_BRIGHT, 1, cv2.LINE_AA)
        cv2.putText(out, "10.0 MHz", (img_x0 + 95, img_y0 + 22),
                    font, 0.35, T_DIM, 1, cv2.LINE_AA)
        cv2.putText(out, "DEPTH 8cm", (img_x0 + 195, img_y0 + 22),
                    font, 0.35, T_DIM, 1, cv2.LINE_AA)

        # SEQ A / B — top right
        cv2.putText(out, "SEQ", (img_x1 - 92, img_y0 + 22),
                    font, 0.30, T_DIM, 1, cv2.LINE_AA)
        col_a = T_BRIGHT if self.active == 0 else T_DIM
        col_b = T_BRIGHT if self.active == 1 else T_DIM
        cv2.putText(out, "A", (img_x1 - 58, img_y0 + 22),
                    font_m, 0.52, col_a, 1, cv2.LINE_AA)
        cv2.putText(out, "B", (img_x1 - 30, img_y0 + 22),
                    font_m, 0.52, col_b, 1, cv2.LINE_AA)

        # ── DEPTH SCALE — left edge ───────────────────────────────────────────
        cone_y0 = img_y0 + int(ih * 0.08)
        cone_y1 = img_y0 + int(ih * 0.96)
        cone_h  = cone_y1 - cone_y0
        for cm in DEPTH_TICKS:
            ty = cone_y0 + int((cm / DEPTH_CM) * cone_h)
            cv2.line(out, (img_x0 + 6, ty), (img_x0 + 14, ty), T_MID, 1)
            cv2.putText(out, str(cm), (img_x0 - 12, ty + 4),
                        font, 0.28, T_MID, 1, cv2.LINE_AA)

        # ── GESTURE STATE — right of image ────────────────────────────────────
        gs_x = img_x1 + 12
        gs_y = img_y1 - 68

        scrub_active = (gesture == "scrub")
        fist_active  = (gesture == "fist")
        sw_active    = fist_active or just_switched

        cv2.putText(out, "GESTURE", (gs_x, gs_y),
                    font, 0.26, T_DIM, 1, cv2.LINE_AA)
        cv2.putText(out, "SCRUB", (gs_x, gs_y + 15),
                    font, 0.40, T_BRIGHT if scrub_active else T_DIM,
                    1, cv2.LINE_AA)

        cv2.putText(out, "HOLD FIST", (gs_x, gs_y + 38),
                    font, 0.26, T_MID if sw_active else T_DIM, 1, cv2.LINE_AA)
        cv2.putText(out, "SWITCH", (gs_x, gs_y + 53),
                    font, 0.40, T_BRIGHT if sw_active else T_DIM,
                    1, cv2.LINE_AA)

        # fist hold progress bar
        if fist_active:
            progress = min(self.fist_count / FIST_HOLD, 1.0)
            bar_h  = 40
            bar_y1 = gs_y + 100
            bar_y0 = bar_y1 - bar_h
            filled = int(bar_h * progress)
            cv2.rectangle(out, (gs_x, bar_y0), (gs_x + 3, bar_y1), T_DARK, -1)
            cv2.rectangle(out, (gs_x, bar_y1 - filled),
                          (gs_x + 3, bar_y1), T_BRIGHT, -1)

        # ── FRAME COUNTER ─────────────────────────────────────────────────────
        cv2.putText(out, f"{fi + 1:03d} / {n:03d}",
                    (img_x1 - 68, img_y1 - 10),
                    font, 0.30, T_DIM, 1, cv2.LINE_AA)

        # ── SCRUB BAR — overlaid at bottom of image ───────────────────────────
        bar_x0     = img_x0 + 10
        bar_x1     = img_x1 - 10
        bar_y      = img_y1 - 20
        bar_w      = bar_x1 - bar_x0
        progress_x = bar_x0 + int((fi / max(n - 1, 1)) * bar_w)

        cv2.rectangle(out, (bar_x0, bar_y - 1), (bar_x1, bar_y + 1), T_DARK, -1)
        cv2.rectangle(out, (bar_x0, bar_y - 1), (progress_x, bar_y + 1), T_MID, -1)
        cv2.circle(out, (progress_x, bar_y), 4, T_BRIGHT, -1)

        # ── BOTTOM LABEL ──────────────────────────────────────────────────────
        label_y = img_y1 + 18
        cv2.putText(out, "LUMBAR SPINE  ·  GESTURESCROLL",
                    (img_x0, label_y),
                    font, 0.28, T_DIM, 1, cv2.LINE_AA)
        cv2.circle(out, (img_x1 - 6, label_y - 4), 3, T_BRIGHT, -1)
        cv2.putText(out, "TRACKING", (img_x1 - 66, label_y),
                    font, 0.26, T_MID, 1, cv2.LINE_AA)

        # ── SWITCH FLASH ─────────────────────────────────────────────────────
        if just_switched:
            cv2.putText(out, f"SEQUENCE {'B' if self.active == 1 else 'A'}",
                        (W // 2 - 60, img_y0 + 52),
                        font_m, 0.72, T_BRIGHT, 1, cv2.LINE_AA)

        return out

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("GestureScroll", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GestureScroll", DISPLAY_W, DISPLAY_H)

        print("GestureScroll running.")
        print("  Open hand + swipe  →  scrub frames")
        print("  Hold fist          →  switch sequence (~0.6s)")
        print("  [ / ]              →  calibrate swipe range")
        print("  A / B              →  jump to sequence")
        print("  , / .              →  step one frame")
        print("  Q / ESC            →  quit\n")

        flash_timer = 0

        while True:
            ret, cam = cap.read()
            if not ret:
                break

            cam  = cv2.flip(cam, 1)
            rgb  = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect_for_video(mp_img, self._timestamp_ms())

            gesture       = "none"
            just_switched = False

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                if self._is_fist(lm):
                    gesture = "fist"
                    self.fist_count += 1
                    self.last_wx = None
                    self.wrist_buf.clear()

                    now = time.time()
                    if (self.fist_count >= FIST_HOLD
                            and now - self.last_switch > FIST_COOLDOWN):
                        self.active      = 1 - self.active
                        self.frame_f     = 0.0
                        self.target      = 0.0
                        self.last_switch = now
                        self.fist_count  = 0
                        flash_timer      = 60
                        just_switched    = True
                else:
                    gesture = "scrub"
                    self.fist_count = 0
                    wx = lm[0].x
                    self.last_wx = wx
                    self.wrist_buf.append(wx)

                    smooth_wx = float(np.mean(self.wrist_buf))
                    norm = (smooth_wx - self.x_min) / max(self.x_max - self.x_min, 0.01)
                    norm = float(np.clip(norm, 0.0, 1.0))
                    n = len(self.sets[self.active])
                    self.target = norm * (n - 1)
            else:
                self.fist_count = 0
                self.last_wx    = None
                self.wrist_buf.clear()

            n = len(self.sets[self.active])
            self.frame_f += (self.target - self.frame_f) * LERP_SPEED
            self.frame_f  = float(np.clip(self.frame_f, 0, n - 1))

            if flash_timer > 0:
                flash_timer -= 1
                just_switched = True

            fi     = int(round(self.frame_f))
            img    = self.sets[self.active][fi]
            canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
            dh, dw = img.shape[:2]
            yo     = (DISPLAY_H - dh) // 2
            xo     = (DISPLAY_W - dw) // 2
            canvas[yo:yo + dh, xo:xo + dw] = img

            output = self._draw_hud(canvas, gesture, just_switched)
            cv2.imshow("GestureScroll", output)

            key    = cv2.waitKey(1)
            masked = key & 0xFF
            if masked in (ord('q'), 27):
                break
            elif masked in (ord('a'), ord('1')):
                self.active, self.frame_f, self.target = 0, 0.0, 0.0
            elif masked in (ord('b'), ord('2')):
                self.active, self.frame_f, self.target = 1, 0.0, 0.0
            elif masked == ord('['):
                if self.last_wx is not None:
                    self.x_min = self.last_wx
                    print(f"Left edge set  → x={self.x_min:.3f}")
            elif masked == ord(']'):
                if self.last_wx is not None:
                    self.x_max = self.last_wx
                    print(f"Right edge set → x={self.x_max:.3f}")
            elif masked == ord(','):
                self.target = max(0.0, self.target - 1)
            elif masked == ord('.'):
                n = len(self.sets[self.active])
                self.target = min(float(n - 1), self.target + 1)

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


if __name__ == "__main__":
    set_a = sys.argv[1] if len(sys.argv) > 1 else "set_a"
    set_b = sys.argv[2] if len(sys.argv) > 2 else "set_b"
    GestureScroll(set_a, set_b).run()
