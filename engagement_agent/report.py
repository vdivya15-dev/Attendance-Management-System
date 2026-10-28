"""Session KPIs and downloadable educator report."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import AUDIO_LABELS
from .utils import fmt_time


def compute_kpis(register: pd.DataFrame, tl: pd.DataFrame, suggestions: List[dict], window_s: float,
                 low_threshold: float, duration: float) -> dict:
    k: dict = {"duration": duration, "windows": len(tl)}
    det = register[~register["Student ID"].astype(str).str.startswith("Roster-")] if not register.empty else register
    k["detected"] = int(len(det))
    k["roster"] = int(len(register))
    k["present"] = int((register["Status"] == "Present").sum()) if not register.empty else 0
    k["attendance_rate"] = (100 * k["present"] / k["roster"]) if k["roster"] else None
    if not tl.empty and tl["engagement"].notna().any():
        e = tl["engagement"]
        k["mean_eng"] = float(e.mean())
        ip, iv = int(e.idxmax()), int(e.idxmin())
        k["peak"], k["peak_t"] = float(e.iloc[ip]), float(tl["t"].iloc[ip])
        k["low"], k["low_t"] = float(e.iloc[iv]), float(tl["t"].iloc[iv])
        k["pct_engaged"] = float(100 * (e >= 60).mean())
        k["pct_low"] = float(100 * (e < low_threshold).mean())
        k["forward_pct"] = float(100 * tl["forward"].mean()) if tl["forward"].notna().any() else None
        k["upright_pct"] = float(100 * tl["upright"].mean()) if tl["upright"].notna().any() else None
        lab = tl["audio_label"]
        valid = lab[lab != "-"]
        k["audio_mix"] = {l: float(100 * (valid == l).mean()) for l in AUDIO_LABELS} if len(valid) else {}
        k["participation_rate"] = k["audio_mix"].get("Student Discussion") if k["audio_mix"] else None
    k["prompts"] = len([s for s in suggestions if s["severity"] in ("warning", "info")])
    return k


def _f(v, suffix="%", nd=0):
    return "n/a" if v is None else f"{v:.{nd}f}{suffix}"


def build_markdown(k: dict, register: pd.DataFrame, suggestions: List[dict], presence_threshold: float,
                   low_threshold: float) -> str:
    L = ["# Session Summary Report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
         "> Privacy note: processed locally; metrics are collective and intended for teaching improvement, "
         "not individual monitoring.", "", "## Key indicators",
         f"- Session length: **{fmt_time(k['duration'])}**",
         f"- Attendance (sustained presence >= {presence_threshold * 100:.0f}%): "
         f"**{k['present']} / {k['roster']}** ({_f(k.get('attendance_rate'))})"]
    if "mean_eng" in k:
        L += [f"- Mean Engagement Index: **{k['mean_eng']:.0f} / 100**",
              f"- Peak: **{k['peak']:.0f}** at {fmt_time(k['peak_t'])}; lowest: **{k['low']:.0f}** at {fmt_time(k['low_t'])}",
              f"- Time with engagement >= 60: **{k['pct_engaged']:.0f}%**; below {low_threshold:.0f}: **{k['pct_low']:.0f}%**",
              f"- Forward-facing attention (class average): {_f(k.get('forward_pct'))}; upright posture: {_f(k.get('upright_pct'))}",
              f"- Verbal participation (share of time in student discussion): {_f(k.get('participation_rate'))}"]
        if k.get("audio_mix"):
            L += ["", "## Acoustic mix"] + [f"- {a}: {v:.0f}%" for a, v in k["audio_mix"].items()]
    L += ["", "## Teaching suggestions"]
    L += ([f"- [{s['time']}] **{s['title']}** - {s['message']}" for s in suggestions] or ["- None triggered."])
    if not register.empty:
        L += ["", "## Attendance register", "", _md_table(register)]
    return "\n".join(L) + "\n"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
