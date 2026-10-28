"""Audio stream processor: framing, spectral features, VAD and acoustic classification."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import AUDIO_SCORE


@dataclass
class AudioResult:
    sr: int
    hop_s: float
    db: np.ndarray
    zcr: np.ndarray
    centroid: np.ndarray
    flat: np.ndarray
    floor_db: float
    ref_db: float
    duration: float
    source: str = "Audio"

    def windows(self, window_s: float = 5.0, sensitivity_db: float = 10.0) -> pd.DataFrame:
        """Aggregate frame features into time windows and classify each window."""
        floor = self.floor_db
        speech = (self.db > floor + sensitivity_db) & (self.flat < 0.30) & (self.zcr < 0.35)
        # bridge short intra-word gaps (<~0.3 s) so only real pauses / turn changes count
        speech = ndimage.binary_closing(speech, structure=np.ones(max(1, int(0.3 / self.hop_s))))
        loud = (self.db > floor + max(sensitivity_db, 12.0)) & ~speech
        fpw = max(1, int(round(window_s / self.hop_s)))
        n = int(math.ceil(len(self.db) / fpw))
        rows = []
        for i in range(n):
            sl = slice(i * fpw, (i + 1) * fpw)
            sp, ld, d = speech[sl], loud[sl], self.db[sl]
            speech_ratio, loud_ratio = float(sp.mean()), float(ld.mean())
            turn_rate = float(np.count_nonzero(sp[1:] != sp[:-1]) / (len(sp) * self.hop_s)) if len(sp) > 1 else 0.0
            c = self.centroid[sl][sp]
            cv = float(c.std() / c.mean()) if len(c) >= 5 and c.mean() > 0 else 0.0
            if loud_ratio >= 0.35:
                label = "High Ambient Distraction"
            elif speech_ratio < 0.08:
                label = "Silence"
            else:
                votes = int(cv > 0.22) + int(turn_rate > 0.35) + int(speech_ratio < 0.80)
                label = "Student Discussion" if votes >= 2 else "Instructor Lecture"
            energy = float(np.clip((d.mean() - floor) / max(1.0, self.ref_db - floor), 0, 1))
            rows.append({"window": i, "t": i * window_s, "label": label,
                         "speech_ratio": round(speech_ratio, 3), "loud_ratio": round(loud_ratio, 3),
                         "turn_rate": round(turn_rate, 3), "energy": round(energy, 3),
                         "audio_score": AUDIO_SCORE[label] * (0.8 + 0.2 * energy)})
        return pd.DataFrame(rows)


def load_wav(data: bytes, max_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Read WAV bytes -> mono float32 in [-1, 1], downsampled to <= max_sr."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    try:
        sr, y = wavfile.read(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("Could not read audio. Please provide an uncompressed WAV file "
                         "(convert MP3/M4A to WAV first).") from exc
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
    else:
        y = y.astype(np.float32)
    if sr > max_sr:
        g = math.gcd(max_sr, sr)
        y = resample_poly(y, max_sr // g, sr // g).astype(np.float32)
        sr = max_sr
    return y.astype(np.float32), int(sr)


def analyze_audio(y: np.ndarray, sr: int, hop_s: float = 0.03, source: str = "Audio") -> AudioResult:
    y = np.asarray(y, dtype=np.float32)
    fl = int(sr * hop_s)
    n = len(y) // fl
    if n < 10:
        raise ValueError("Audio is too short to analyse (need at least a few seconds).")
    y = y[: n * fl]
    win = np.hanning(fl).astype(np.float32)
    freqs = np.fft.rfftfreq(fl, 1.0 / sr)
    band = (freqs >= 100) & (freqs <= min(4000, sr / 2))
    fb = freqs[band]
    db, zcr = np.empty(n), np.empty(n)
    cen, flat = np.empty(n), np.empty(n)
    CH = 4000
    for s in range(0, n, CH):
        e = min(n, s + CH)
        fr = y[s * fl:e * fl].reshape(-1, fl)
        db[s:e] = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
        zcr[s:e] = np.mean(fr[:, 1:] * fr[:, :-1] < 0, axis=1)
        mag = np.abs(np.fft.rfft(fr * win, axis=1))[:, band] + 1e-9
        cen[s:e] = (mag * fb).sum(1) / mag.sum(1)
        flat[s:e] = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    floor = max(float(np.percentile(db, 5)), -70.0)
    ref = float(np.percentile(db, 95))
    return AudioResult(sr, hop_s, db, zcr, cen, flat, floor, max(ref, floor + 1), n * hop_s, source)
