# 🎓 Multimodal Attendance & Engagement Agent
Skill Wallet Capstone – Project 5. A privacy-first Streamlit dashboard that turns classroom video + audio into
presence-based attendance, acoustic dynamics, a fused Engagement Index, teaching suggestions and an educator report.

## Features
| Tab | What it does |
|---|---|
| 1 · Attendance | Webcam snapshots / uploaded video / photos → face detection (OpenCV Haar, optional MediaPipe head pose) → anonymous tracking IDs → **sustained-presence rule** (default ≥70% of session) → CSV register |
| 2 · Acoustic Dynamics | WAV upload or mic recording → framed spectral energy, ZCR, spectral flatness + VAD → **Silence / Instructor Lecture / Student Discussion / High Ambient Distraction** per window |
| 3 · Engagement Fusion | Visual attentiveness (forward head pose, upright vs slumped posture) + audio interaction energy → time-step matrix → **Engagement Index 0-100**, peaks/dips, replay slider |
| 4 · Suggestions | Rule-based advisor (sustained low engagement, long one-way lecture, noise, sudden drop, recovery, attendance) |
| 5 · Report | KPIs, participation rate, charts, downloadable Markdown report + CSVs |

No database, no API keys, no cloud AI calls. **Click "Load demo session" in the sidebar** to explore instantly.

## Run locally (recommended for real classrooms – true edge processing)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional, better head pose (Python 3.9-3.12): pip install -r requirements-optional.txt
streamlit run app.py
```

## Push to GitHub
```bash
cd attendance-engagement-agent
git init && git add . && git commit -m "Multimodal Attendance & Engagement Agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud
1. share.streamlit.io → **New app** → pick your repo, branch `main`, main file `app.py` → Deploy.
2. Dependencies install automatically from `requirements.txt`.

> ⚠️ A hosted deployment processes uploads/snapshots **on Streamlit's server**. That is fine for demos and
> synthetic/consented data, but for real classrooms run it locally so video never leaves the machine.

## Project structure
```
app.py                      Streamlit UI (5 tabs)
engagement_agent/
  vision.py                 detection, tracking, attentiveness, attendance register
  audio.py                  framing, features, VAD, window classification
  fusion.py                 time-step matrix + Engagement Index, peaks/dips
  advisor.py                heuristic teaching-suggestion engine
  report.py                 KPIs + Markdown report
  charts.py / demo.py / config.py / utils.py
```

## How the scores work
- **Presence**: samples where a tracked face is seen ÷ session length; Present if ≥ threshold.
- **Visual attentiveness** per face = 0.6·forward-facing + 0.4·upright (only class averages are shown).
- **Audio score** = class weight (Discussion 1.0, Lecture 0.6, Silence 0.25, Distraction 0) × (0.8 + 0.2·energy).
- **Engagement Index** = 100 · (w·visual + (1−w)·audio), lightly smoothed. Falls back to whichever modality exists.

## Known limitations (be upfront in your report)
- Heuristic, not a trained model: Haar cascades work best with front-facing, well-lit faces; crowded or
  low-resolution rooms will fragment IDs. Tune sampling rate, thresholds and expected class size in the sidebar.
- Audio classes are rule-based; lecture-vs-discussion separation is approximate on real recordings. Use WAV input.
- Anonymous IDs (S01…) are not identities; there is no face recognition or storage of faces.
