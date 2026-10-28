"""Matplotlib charts with a clean dashboard look."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .config import AUDIO_COLORS, AUDIO_LABELS

BLUE, RED, GREEN, GREY = "#2563EB", "#EF4444", "#10B981", "#94A3B8"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def engagement_fig(tl: pd.DataFrame, threshold: float, peaks, dips, cursor: int | None = None):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = tl["t"] / 60.0
    y = tl["engagement"]
    k = len(tl) - 1 if cursor is None else int(cursor)
    ax.plot(x, y, color="#CBD5E1", lw=1.5)
    ax.plot(x[:k + 1], y[:k + 1], color=BLUE, lw=2.4)
    ax.fill_between(x[:k + 1], y[:k + 1], threshold, where=(y[:k + 1] < threshold), color=RED, alpha=0.18,
                    interpolate=True)
    ax.axhline(threshold, color=RED, ls="--", lw=1, label=f"Low-engagement threshold ({threshold:.0f})")
    pk = [i for i in peaks if i <= k]
    dp = [i for i in dips if i <= k]
    ax.scatter(x.iloc[pk], y.iloc[pk], color=GREEN, zorder=5, s=36, label="Attention peak")
    ax.scatter(x.iloc[dp], y.iloc[dp], color=RED, zorder=5, s=36, label="Attention dip")
    ax.scatter([x.iloc[k]], [y.iloc[k]], color=BLUE, s=60, zorder=6, edgecolor="white")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(x.iloc[-1], 0.1))
    ax.set_xlabel("Session time (minutes)")
    ax.set_ylabel("Engagement Index")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def audio_fig(aw: pd.DataFrame, window_s: float):
    fig, ax = plt.subplots(figsize=(10, 3.0))
    colors = [AUDIO_COLORS[l] for l in aw["label"]]
    ax.bar(aw["t"] / 60.0, aw["energy"].clip(lower=0.04), width=window_s / 60.0, align="edge", color=colors)
    ax.set_xlabel("Session time (minutes)")
    ax.set_ylabel("Relative energy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, aw["t"].iloc[-1] / 60.0 + window_s / 60.0)
    ax.legend(handles=[Patch(color=AUDIO_COLORS[l], label=l) for l in AUDIO_LABELS], ncol=4, fontsize=8,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    _style(ax)
    fig.tight_layout()
    return fig


def mix_fig(mix: dict):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    labs = [l for l in AUDIO_LABELS if l in mix]
    vals = [mix[l] for l in labs]
    ax.barh(labs[::-1], vals[::-1], color=[AUDIO_COLORS[l] for l in labs[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of session time")
    _style(ax)
    fig.tight_layout()
    return fig


def attendance_fig(reg: pd.DataFrame, threshold: float):
    d = reg.sort_values("Student ID")
    fig, ax = plt.subplots(figsize=(5, max(2.5, 0.28 * len(d) + 0.8)))
    ax.barh(d["Student ID"][::-1], d["Presence (%)"][::-1],
            color=[GREEN if s == "Present" else GREY for s in d["Status"][::-1]])
    ax.axvline(threshold * 100, color=RED, ls="--", lw=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Presence % (required: {threshold * 100:.0f}%)")
    _style(ax)
    fig.tight_layout()
    return fig
