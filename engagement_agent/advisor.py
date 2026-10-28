"""Heuristic teaching-suggestion engine. Advice is about the class as a whole, never individuals."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import fmt_time


def _runs(mask: np.ndarray):
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


def _mode(s: pd.Series) -> str:
    s = s[s != "-"]
    return s.mode().iloc[0] if len(s) else "-"


def generate_suggestions(tl: pd.DataFrame, window_s: float, threshold: float = 45.0,
                         sustained_min: float = 5.0, lecture_max_min: float = 7.0,
                         register: Optional[pd.DataFrame] = None) -> List[dict]:
    out: List[dict] = []
    if tl.empty or tl["engagement"].isna().all():
        return out
    eng = tl["engagement"].to_numpy()
    labels = tl["audio_label"]
    n = len(tl)
    need = max(1, int(round(sustained_min * 60 / window_s)))

    # 1) Sustained low engagement (+ recovery follow-up)
    for i, j in _runs(eng < threshold):
        if j - i + 1 < need:
            continue
        seg = tl.iloc[i:j + 1]
        dom = _mode(seg["audio_label"])
        action = {
            "Instructor Lecture": "recommend launching a quick poll, interactive discussion, or a 2-minute stretch break",
            "Silence": "recommend asking an open question or a short think-pair-share",
            "High Ambient Distraction": "classroom noise is high - recommend a quiet-signal reset followed by a short group task",
            "Student Discussion": "discussion is active but attention is low - recommend a brief recap or one focusing question",
        }.get(dom, "recommend an interactive activity or a short break")
        if seg["upright"].mean() < 0.5:
            action += " (posture indicators also suggest fatigue)"
        dur = f"{sustained_min:g} minute{'s' if sustained_min != 1 else ''}"
        out.append({"t": (i + need) * window_s, "severity": "warning", "rule": "sustained_low",
                    "title": "Sustained low collective engagement",
                    "message": f"Collective engagement dropped below {threshold:.0f}% for {dur} - {action}."})
        if j + 1 < n and eng[j + 1] >= threshold:
            out.append({"t": (j + 1) * window_s, "severity": "success", "rule": "recovery",
                        "title": "Engagement recovered",
                        "message": f"Collective engagement is back above {threshold:.0f}% "
                                   f"({eng[j + 1]:.0f}%) - the recent change of activity appears to be working."})

    # 2) Long stretch of one-way delivery
    not_disc = (labels != "Student Discussion").to_numpy()
    need_l = max(1, int(round(lecture_max_min * 60 / window_s)))
    for i, j in _runs(not_disc):
        if j - i + 1 >= need_l:
            out.append({"t": (i + need_l) * window_s, "severity": "info", "rule": "long_lecture",
                        "title": "Long stretch without student interaction",
                        "message": f"No student discussion detected for {lecture_max_min:g} minutes - "
                                   "consider a check-for-understanding question or a pair discussion."})

    # 3) Sustained ambient distraction (>= 30 s)
    need_d = max(1, int(round(30 / window_s)))
    for i, j in _runs((labels == "High Ambient Distraction").to_numpy()):
        if j - i + 1 >= need_d:
            out.append({"t": (i + need_d) * window_s, "severity": "warning", "rule": "noise",
                        "title": "High ambient distraction",
                        "message": "Sustained classroom noise detected - try a quiet signal or restate "
                                   "the task, then give a short, clearly structured activity."})

    # 4) Sudden drop (>= 25 points within ~1 minute), with a 3-minute cooldown
    k = max(2, int(round(60 / window_s)))
    last = -1e9
    for i in range(k, n):
        if eng[i - k:i].max() - eng[i] >= 25 and (i * window_s - last) >= 180:
            last = i * window_s
            out.append({"t": i * window_s, "severity": "info", "rule": "sudden_drop",
                        "title": "Sudden engagement drop",
                        "message": "Engagement fell sharply within about a minute - a brief pause, "
                                   "re-question or change of pace may help."})

    # 5) Attendance context
    if register is not None and not register.empty:
        det = register[~register["Student ID"].str.startswith("Roster-")]
        if len(det) >= 3 and (det["Status"] == "Present").mean() < 0.6:
            out.append({"t": 0.0, "severity": "info", "rule": "attendance",
                        "title": "Low sustained attendance",
                        "message": "Fewer than 60% of detected participants stayed for the required share "
                                   "of the session - consider checking camera coverage or session timing."})

    # 6) Positive reinforcement
    if not any(o["severity"] == "warning" for o in out) and np.nanmean(eng) >= 70:
        out.append({"t": float(n * window_s), "severity": "success", "rule": "strong",
                    "title": "Strong collective engagement",
                    "message": "Engagement stayed high throughout - the current pacing and activity mix are working well."})

    out.sort(key=lambda o: o["t"])
    for o in out:
        o["time"] = fmt_time(o["t"])
    return out
