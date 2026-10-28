"""Synthetic demo session so the dashboard can be explored with no camera/microphone/files."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .audio import AudioResult, analyze_audio
from .vision import OBS_COLUMNS, VisionResult

SR = 8000
DURATION = 720
SCHEDULE = [(0, 100, "lecture"), (100, 130, "discussion"), (130, 300, "lecture"), (300, 330, "distraction"),
            (330, 420, "silence"), (420, 450, "distraction"), (450, 570, "lecture"),
            (570, 640, "discussion"), (640, 680, "lecture"), (680, 720, "discussion")]


def _voice(n: int, f0: float, rng, amp: float = 0.2) -> np.ndarray:
    t = np.arange(n) / SR
    f = f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    sig = np.zeros(n)
    for h in range(1, 13):
        sig += np.sin(h * phase) / h ** 1.1
    env = np.clip(0.8 * np.sin(2 * np.pi * 3.2 * t + rng.uniform(0, 6)) + 0.55, 0, 1)
    return (amp * sig / (np.abs(sig).max() + 1e-9) * env).astype(np.float32)


def make_demo_audio(seed: int = 7) -> AudioResult:
    rng = np.random.default_rng(seed)
    y = (rng.normal(0, 0.003, DURATION * SR)).astype(np.float32)
    for a, b, kind in SCHEDULE:
        s, e = a * SR, b * SR
        n = e - s
        if kind == "lecture":
            y[s:e] += _voice(n, 115, rng)
        elif kind == "discussion":
            pos = s
            while pos < e:
                L = int(rng.uniform(1.2, 3.0) * SR)
                L = min(L, e - pos)
                y[pos:pos + L] += _voice(L, float(rng.choice([95, 150, 190, 230, 260])), rng, amp=0.18)
                pos += L + int(rng.uniform(0.3, 0.7) * SR)
        elif kind == "distraction":
            y[s:e] += rng.normal(0, 0.12, n).astype(np.float32)
    return analyze_audio(y, SR, source="Demo (synthetic)")


def make_demo_vision(seed: int = 11, n_students: int = 14, dt: float = 2.0) -> VisionResult:
    rng = np.random.default_rng(seed)
    ts = np.arange(0, DURATION, dt)
    knots_t = [0, 200, 235, 565, 600, DURATION]
    knots_p = [0.88, 0.88, 0.30, 0.30, 0.84, 0.88]
    p_t = np.interp(ts, knots_t, knots_p)
    spans = {i: (0, DURATION) for i in range(n_students)}
    spans[10] = (260, DURATION)   # late arrival -> Absent (64%)
    spans[11] = (150, DURATION)   # late but sustained -> Present (79%)
    spans[12] = (310, 360)        # hallway pass-through -> Absent
    spans[13] = (0, 380)          # leaves early -> Absent (53%)
    rows = []
    for i in range(n_students):
        a, b = spans[i]
        for k, t in enumerate(ts):
            if not (a <= t < b) or rng.random() < 0.04:
                continue
            p = p_t[k]
            fwd = int(rng.random() < p)
            up = int(rng.random() < min(1.0, p + 0.1))
            rows.append({"t": float(t), "student": f"S{i + 1:02d}", "x": 80 + (i % 5) * 170,
                         "y": 90 + (i // 5) * 140, "w": 60, "h": 60, "forward": fwd, "upright": up,
                         "attn": 0.6 * fwd + 0.4 * up})
    df = pd.DataFrame(rows, columns=OBS_COLUMNS)
    canvas = np.full((420, 960, 3), 245, np.uint8)
    for i in range(n_students):
        x, y = 80 + (i % 5) * 170, 90 + (i // 5) * 140
        cv2.rectangle(canvas, (x, y), (x + 60, y + 60), (235, 99, 37), 2)
        cv2.putText(canvas, f"S{i + 1:02d}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 99, 37), 2)
    cv2.putText(canvas, "Synthetic demo - no real people or video", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    return VisionResult(df, float(DURATION), dt, len(ts), canvas, "Demo (synthetic)", "Synthetic")
