"""Multimodal fusion: visual attentiveness + audio interaction -> Engagement Index (0-100)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .audio import AudioResult
from .vision import VisionResult

TL_COLUMNS = ["window", "t", "visual", "forward", "upright", "faces", "audio_label",
              "audio_score", "engagement_raw", "engagement"]


def build_timeline(vision: Optional[VisionResult], audio: Optional[AudioResult], window_s: float = 5.0,
                   w_visual: float = 0.65, smooth: int = 3, sensitivity_db: float = 10.0
                   ) -> Tuple[pd.DataFrame, str]:
    """Return (time-step matrix, mode) where mode in {fused, visual-only, audio-only, none}."""
    has_v_data = vision is not None and not vision.obs.empty
    durations = []
    if vision is not None:
        durations.append(vision.duration)
    if audio is not None:
        durations.append(audio.duration)
    if not durations or max(durations) <= 0:
        return pd.DataFrame(columns=TL_COLUMNS), "none"
    n = max(1, int(math.ceil(max(durations) / window_s)))
    tl = pd.DataFrame({"window": range(n)})
    tl["t"] = tl["window"] * window_s
    for c in ("visual", "forward", "upright", "audio_score"):
        tl[c] = np.nan
    tl["faces"] = 0
    tl["audio_label"] = "-"

    if has_v_data:
        o = vision.obs.assign(window=(vision.obs["t"] // window_s).astype(int))
        agg = o.groupby("window").agg(visual=("attn", "mean"), forward=("forward", "mean"),
                                      upright=("upright", "mean"), faces=("student", "nunique"))
        agg = agg.reindex(range(n))
        for c in ("visual", "forward", "upright"):
            tl[c] = agg[c].to_numpy()
        tl["faces"] = agg["faces"].fillna(0).astype(int).to_numpy()
    if audio is not None:
        aw = audio.windows(window_s, sensitivity_db).set_index("window").reindex(range(n))
        tl["audio_score"] = aw["audio_score"].to_numpy()
        tl["audio_label"] = aw["label"].fillna("-").to_numpy()

    has_v = tl["visual"].notna().any()
    has_a = tl["audio_score"].notna().any()
    vi = tl["visual"].interpolate(limit_direction="both") if has_v else None
    ai = tl["audio_score"].interpolate(limit_direction="both") if has_a else None
    if has_v and has_a:
        raw, mode = w_visual * vi + (1 - w_visual) * ai, "fused"
    elif has_v:
        raw, mode = vi, "visual-only"
    elif has_a:
        raw, mode = ai, "audio-only"
    else:
        raw, mode = pd.Series(np.nan, index=tl.index), "none"
    tl["engagement_raw"] = (100 * raw).clip(0, 100)
    tl["engagement"] = tl["engagement_raw"].rolling(max(1, smooth), center=True, min_periods=1).mean()
    return tl[TL_COLUMNS], mode


def find_extremes(tl: pd.DataFrame, prominence: float = 8.0):
    """Indices of attention peaks and dips on the smoothed engagement curve."""
    if tl.empty or tl["engagement"].isna().all() or len(tl) < 3:
        return [], []
    y = tl["engagement"].to_numpy()
    peaks, _ = find_peaks(y, prominence=prominence, distance=2)
    dips, _ = find_peaks(-y, prominence=prominence, distance=2)
    return peaks.tolist(), dips.tolist()
