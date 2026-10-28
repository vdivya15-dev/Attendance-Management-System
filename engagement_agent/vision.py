"""Face detection, ID tracking, attentiveness indicators and presence-based attendance.

All processing happens in memory on the machine running the app. No frames are
written to disk and nothing is sent to any external service.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .utils import fmt_time

MAX_WIDTH = 960
OBS_COLUMNS = ["t", "student", "x", "y", "w", "h", "forward", "upright", "attn"]


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #
@dataclass
class VisionResult:
    obs: pd.DataFrame            # one row per (sample time, tracked face)
    duration: float              # session length in seconds
    dt: float                    # seconds represented by one sample
    n_samples: int
    preview: Optional[np.ndarray]  # RGB annotated preview (faces blurred by default)
    source: str
    backend: str = "OpenCV Haar"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_inside(a, b) -> bool:
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]


class FaceDetector:
    """Haar-cascade face detector with optional MediaPipe FaceMesh head pose."""

    def __init__(self, use_mediapipe: bool = False,
                 yaw_range=(0.36, 0.64), pitch_max: float = 0.72):
        base = cv2.data.haarcascades
        self.front = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
        self.profile = cv2.CascadeClassifier(os.path.join(base, "haarcascade_profileface.xml"))
        if self.front.empty():
            raise RuntimeError("OpenCV Haar cascade for faces could not be loaded.")
        self.yaw_range, self.pitch_max = yaw_range, pitch_max
        self.mesh = None
        self.backend = "OpenCV Haar"
        if use_mediapipe:
            try:
                import mediapipe as mp  # optional dependency
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.4)
                self.backend = "OpenCV + MediaPipe FaceMesh"
            except Exception:
                self.mesh = None

    def _pose(self, frame, box):
        """Return (forward, looking_down) for one frontal face."""
        if self.mesh is None:
            return True, False
        try:
            x, y, w, h = box
            pad = int(0.25 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            res = self.mesh.process(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                nose, left, right, top, chin = lm[1], lm[234], lm[454], lm[10], lm[152]
                span = right.x - left.x
                pspan = chin.y - top.y
                yaw = (nose.x - left.x) / span if abs(span) > 1e-6 else 0.5
                pitch = (nose.y - top.y) / pspan if abs(pspan) > 1e-6 else 0.5
                return self.yaw_range[0] <= yaw <= self.yaw_range[1], pitch > self.pitch_max
        except Exception:
            pass
        return True, False

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        _, W = gray.shape
        ms = max(24, int(0.03 * W))
        faces: List[dict] = []
        for (x, y, w, h) in self.front.detectMultiScale(gray, 1.1, 5, minSize=(ms, ms)):
            box = (int(x), int(y), int(w), int(h))
            fwd, down = self._pose(frame_bgr, box)
            faces.append({"box": box, "forward": fwd, "down": down})
        if not self.profile.empty():  # side-on faces = head turned away
            cands = [tuple(map(int, b)) for b in self.profile.detectMultiScale(gray, 1.1, 6, minSize=(ms, ms))]
            flipped = cv2.flip(gray, 1)
            for (x, y, w, h) in self.profile.detectMultiScale(flipped, 1.1, 6, minSize=(ms, ms)):
                cands.append((int(W - x - w), int(y), int(w), int(h)))
            for b in cands:
                if all(_iou(b, f["box"]) < 0.15 and not _center_inside(b, f["box"])
                       and not _center_inside(f["box"], b) for f in faces):
                    faces.append({"box": b, "forward": False, "down": False})
        return faces


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
class CentroidTracker:
    """Greedy nearest-centroid tracker that keeps IDs through short dropouts."""

    def __init__(self, max_missing: int = 8):
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}

    def update(self, boxes) -> List[int]:
        ids: List[Optional[int]] = [None] * len(boxes)
        pairs = []
        for i, (x, y, w, h) in enumerate(boxes):
            cx, cy = x + w / 2, y + h / 2
            for tid, t in self.tracks.items():
                d = math.hypot(cx - t["cx"], cy - t["cy"])
                if d <= 1.0 * max(w, t["w"]):
                    pairs.append((d, i, tid))
        pairs.sort()
        used_b, used_t = set(), set()
        for d, i, tid in pairs:
            if i in used_b or tid in used_t:
                continue
            used_b.add(i)
            used_t.add(tid)
            x, y, w, h = boxes[i]
            self.tracks[tid] = {"cx": x + w / 2, "cy": y + h / 2, "w": w, "missing": 0}
            ids[i] = tid
        for i, (x, y, w, h) in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.tracks[self.next_id] = {"cx": x + w / 2, "cy": y + h / 2, "w": w, "missing": 0}
                self.next_id += 1
        for tid in list(self.tracks):
            if tid not in used_t and tid not in ids:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        return ids  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Session analysis
# --------------------------------------------------------------------------- #
def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        s = MAX_WIDTH / w
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    return frame


def _annotate(frame, items, blur: bool) -> np.ndarray:
    out = frame.copy()
    for label, (x, y, w, h) in items:
        if blur:
            roi = out[max(0, y):y + h, max(0, x):x + w]
            if roi.size:
                out[max(0, y):y + h, max(0, x):x + w] = cv2.GaussianBlur(roi, (0, 0), max(6, w / 6))
        cv2.rectangle(out, (x, y), (x + w, y + h), (37, 99, 235), 2)
        cv2.putText(out, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (37, 99, 235), 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class SessionAnalyzer:
    def __init__(self, dt: float, track_memory_s: float = 5.0,
                 use_mediapipe: bool = False, blur_preview: bool = True, min_missing: int = 1):
        self.dt = dt
        self.det = FaceDetector(use_mediapipe)
        self.tracker = CentroidTracker(max(min_missing, int(round(track_memory_s / dt))))
        self.blur = blur_preview
        self.rows: List[dict] = []
        self.n = 0
        self._best = (0, None, None)

    def feed(self, t: float, frame_bgr: np.ndarray) -> None:
        frame = _resize(frame_bgr)
        faces = self.det.detect(frame)
        ids = self.tracker.update([f["box"] for f in faces])
        for f, tid in zip(faces, ids):
            x, y, w, h = f["box"]
            self.rows.append({"t": float(t), "track": tid, "x": x, "y": y, "w": w, "h": h,
                              "forward": int(f["forward"]), "down": int(f["down"])})
        if len(faces) > self._best[0]:
            self._best = (len(faces), frame.copy(), list(zip(ids, [f["box"] for f in faces])))
        self.n += 1

    def finish(self, source: str) -> VisionResult:
        duration = self.n * self.dt
        df = pd.DataFrame(self.rows)
        if df.empty:
            return VisionResult(pd.DataFrame(columns=OBS_COLUMNS), duration, self.dt, self.n,
                                None, source, self.det.backend)
        min_obs = 3 if self.n >= 10 else (2 if self.n >= 4 else 1)  # drop one-off false positives
        counts = df.groupby("track")["t"].transform("count")
        df = df[counts >= min_obs].copy()
        if df.empty:
            return VisionResult(pd.DataFrame(columns=OBS_COLUMNS), duration, self.dt, self.n,
                                None, source, self.det.backend)
        # Posture: slumping = face drops well below the person's own upright baseline.
        df["cy"] = df["y"] + df["h"] / 2
        base = df.groupby("track")["cy"].transform(lambda s: s.quantile(0.2) if len(s) >= 5 else np.nan)
        slump = ((df["cy"] - base) / df["h"] > 0.45).fillna(False)
        df["upright"] = (~slump & (df["down"] == 0)).astype(int)
        df["attn"] = 0.6 * df["forward"] + 0.4 * df["upright"]
        order = df.groupby("track")["t"].min().sort_values(kind="stable").index
        mapping = {tid: f"S{i + 1:02d}" for i, tid in enumerate(order)}
        df["student"] = df["track"].map(mapping)
        df = df[OBS_COLUMNS].reset_index(drop=True)
        preview = None
        if self._best[1] is not None:
            items = [(mapping[t], b) for t, b in self._best[2] if t in mapping]
            preview = _annotate(self._best[1], items, self.blur)
        return VisionResult(df, duration, self.dt, self.n, preview, source, self.det.backend)


def analyze_video(path: str, sample_fps: float = 2.0, max_seconds: Optional[float] = None,
                  track_memory_s: float = 5.0, use_mediapipe: bool = False, blur_preview: bool = True,
                  progress: Optional[Callable[[float], None]] = None) -> VisionResult:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file. Try an MP4 (H.264) file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    an = SessionAnalyzer(step / fps, track_memory_s, use_mediapipe, blur_preview)
    idx = 0
    try:
        while cap.grab():
            if idx % step == 0:
                t = idx / fps
                if max_seconds and t > max_seconds:
                    break
                ok, frame = cap.retrieve()
                if ok:
                    an.feed(t, frame)
                if progress and total:
                    progress(min(1.0, idx / total))
            idx += 1
    finally:
        cap.release()
    if an.n == 0:
        raise ValueError("No readable frames found in the video.")
    return an.finish("Video")


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode one of the images.")
    return img


def analyze_images(frames: List[np.ndarray], interval_s: float = 30.0, use_mediapipe: bool = False,
                   blur_preview: bool = True, source: str = "Photos",
                   progress: Optional[Callable[[float], None]] = None) -> VisionResult:
    if not frames:
        raise ValueError("No images provided.")
    an = SessionAnalyzer(interval_s, interval_s * 2, use_mediapipe, blur_preview, min_missing=2)
    for i, fr in enumerate(frames):
        an.feed(i * interval_s, fr)
        if progress:
            progress((i + 1) / len(frames))
    return an.finish(source)


# --------------------------------------------------------------------------- #
# Attendance register (sustained-presence rule)
# --------------------------------------------------------------------------- #
REGISTER_COLUMNS = ["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence (%)", "Status"]


def build_register(vr: Optional[VisionResult], threshold: float = 0.70, roster_size: int = 0) -> pd.DataFrame:
    """A participant is Present only if detected for >= threshold of the session."""
    if vr is None or vr.duration <= 0:
        return pd.DataFrame(columns=REGISTER_COLUMNS)
    rows = []
    for sid, d in vr.obs.groupby("student"):
        total = min(len(d) * vr.dt, vr.duration)
        pct = total / vr.duration
        rows.append({
            "Student ID": sid,
            "First Seen": fmt_time(d["t"].min()),
            "Last Seen": fmt_time(d["t"].max() + vr.dt),
            "Total Duration": fmt_time(total),
            "Presence (%)": round(100 * pct, 1),
            "Status": "Present" if pct >= threshold else "Absent",
        })
    extra = max(0, roster_size - len(rows))
    for k in range(extra):
        rows.append({"Student ID": f"Roster-{k + 1:02d}", "First Seen": "-", "Last Seen": "-",
                     "Total Duration": "00:00", "Presence (%)": 0.0, "Status": "Absent"})
    return pd.DataFrame(rows, columns=REGISTER_COLUMNS)
