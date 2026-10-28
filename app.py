"""Multimodal Attendance & Engagement Agent - Streamlit dashboard (Skill Wallet Capstone Project 5)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile

import pandas as pd
import streamlit as st

from engagement_agent import advisor, audio as au, charts, fusion, report, vision as vi
from engagement_agent.config import APP_NAME, AUDIO_COLORS
from engagement_agent.demo import make_demo_audio, make_demo_vision
from engagement_agent.utils import fmt_time

st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {background:#F1F5F9; border-radius:12px; padding:12px 16px;}
.hero {background:linear-gradient(90deg,#2563EB,#7C3AED); color:white; padding:18px 24px; border-radius:14px; margin-bottom:12px;}
.hero h2 {margin:0; color:white;} .hero p {margin:4px 0 0 0; opacity:.9;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _demo_audio():
    return make_demo_audio()


@st.cache_data(show_spinner=False)
def _demo_vision():
    return make_demo_vision()


S = st.session_state
for key, default in (("vision", None), ("audio", None), ("snaps", []), ("last_snap", None)):
    S.setdefault(key, default)
HAS_MP = importlib.util.find_spec("mediapipe") is not None

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🔒 Privacy-first design")
    st.markdown(
        "- Frames and audio are processed **in memory** on the machine running this app\n"
        "- **Nothing is sent to any cloud AI/video service**; no images or audio are saved\n"
        "- Engagement is reported for the **class as a whole**, not per person\n"
        "- Previews blur faces by default")
    st.caption("For true classroom edge use, run locally (`streamlit run app.py`). A hosted deployment "
               "processes uploads on that server, so use it for demos or non-sensitive data.")
    st.divider()
    st.header("⚙️ Settings")
    window_s = st.select_slider("Analysis window (seconds)", [2, 5, 10, 15, 30], value=5)
    presence_pct = st.slider("Sustained-presence threshold (%)", 50, 95, 70)
    roster = st.number_input("Expected class size (0 = auto)", 0, 500, 0)
    w_visual = st.slider("Visual weight in fusion (%)", 0, 100, 65)
    low_thr = st.slider("Low-engagement threshold", 20, 70, 45)
    sustained_min = st.slider("Prompt after low engagement lasts (min)", 0.5, 15.0, 5.0, 0.5)
    lecture_max = st.slider("Max one-way lecture before prompt (min)", 3.0, 20.0, 7.0, 0.5)
    vad_db = st.slider("VAD sensitivity (dB above noise floor)", 4, 20, 10)
    blur = st.checkbox("Blur faces in previews", True)
    use_mp = st.checkbox("Use MediaPipe head pose", False, disabled=not HAS_MP,
                         help="Install requirements-optional.txt to enable.")
    st.divider()
    if st.button("▶ Load demo session", width="stretch"):
        S.vision, S.audio = _demo_vision(), _demo_audio()
        st.rerun()
    if st.button("🗑 Reset session", width="stretch"):
        S.vision, S.audio, S.snaps, S.last_snap = None, None, [], None
        st.rerun()

st.markdown(f'<div class="hero"><h2>🎓 {APP_NAME}</h2>'
            '<p>Presence-based attendance · acoustic dynamics · multimodal engagement · teaching suggestions</p></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ derived data
threshold = presence_pct / 100.0
vr, ar = S.vision, S.audio
register = vi.build_register(vr, threshold, int(roster))
tl, mode = fusion.build_timeline(vr, ar, float(window_s), w_visual / 100.0, sensitivity_db=float(vad_db))
peaks, dips = fusion.find_extremes(tl)
suggestions = advisor.generate_suggestions(tl, float(window_s), float(low_thr), float(sustained_min),
                                           float(lecture_max), register)
duration = max([d for d in (vr.duration if vr else 0, ar.duration if ar else 0)] or [0])
kpis = report.compute_kpis(register, tl, suggestions, float(window_s), float(low_thr), duration)

t1, t2, t3, t4, t5 = st.tabs(["1 · Attendance", "2 · Acoustic Dynamics", "3 · Engagement Fusion",
                              "4 · Teaching Suggestions", "5 · Session Report"])

# ------------------------------------------------------------------ tab 1
with t1:
    st.subheader("Automated presence-based attendance")
    st.caption("Participants are tracked with anonymous IDs. They count as **Present** only if detected for "
               f"at least **{presence_pct}%** of the session, so brief hallway pass-throughs are ignored.")
    src = st.radio("Video source", ["Upload video", "Upload photos", "Webcam snapshots"], horizontal=True)
    try:
        if src == "Upload video":
            f = st.file_uploader("Classroom video", type=["mp4", "avi", "mov", "mkv"])
            c1, c2 = st.columns(2)
            fps = c1.slider("Sampling rate (frames/sec analysed)", 0.5, 5.0, 2.0, 0.5)
            max_s = c2.number_input("Analyse only first N seconds (0 = all)", 0, 36000, 0)
            if f and st.button("Analyze video", type="primary"):
                path = None
                bar = st.progress(0.0, text="Detecting and tracking faces...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                        tmp.write(f.getbuffer())
                        path = tmp.name
                    S.vision = vi.analyze_video(path, fps, max_s or None, 5.0, use_mp, blur,
                                                lambda p: bar.progress(p, text="Detecting and tracking faces..."))
                finally:
                    if path and os.path.exists(path):
                        os.remove(path)
                    bar.empty()
                st.rerun()
        elif src == "Upload photos":
            files = st.file_uploader("Classroom photos (in time order)", type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True)
            interval = st.number_input("Seconds between photos", 1, 3600, 30)
            if files and st.button("Analyze photos", type="primary"):
                frames = [vi.decode_image(x.getvalue()) for x in files]
                S.vision = vi.analyze_images(frames, float(interval), use_mp, blur, "Photos")
                st.rerun()
        else:
            snap = st.camera_input("Capture a classroom snapshot (repeat during the session)")
            if snap is not None:
                data = snap.getvalue()
                h = hashlib.md5(data).hexdigest()
                if S.last_snap != h:
                    S.snaps.append(data)
                    S.last_snap = h
            interval = st.number_input("Treat each snapshot as N seconds of session", 1, 3600, 30)
            c1, c2 = st.columns(2)
            c1.metric("Snapshots collected", len(S.snaps))
            if c2.button("Clear snapshots"):
                S.snaps, S.last_snap = [], None
                st.rerun()
            if len(S.snaps) >= 1 and st.button("Analyze snapshots", type="primary"):
                frames = [vi.decode_image(b) for b in S.snaps]
                S.vision = vi.analyze_images(frames, float(interval), use_mp, blur, "Webcam snapshots")
                st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not process the visual input: {exc}")

    if vr is None:
        st.info("Provide a video/photos, or click **Load demo session** in the sidebar.")
    elif vr.obs.empty:
        st.warning("No faces were detected. Try a clearer, front-facing view or a higher sampling rate.")
    else:
        present = int((register["Status"] == "Present").sum())
        m = st.columns(4)
        m[0].metric("Participants tracked", int(vr.obs["student"].nunique()))
        m[1].metric("Present", f"{present} / {len(register)}")
        m[2].metric("Session length", fmt_time(vr.duration))
        m[3].metric("Detector", vr.backend)
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(register, width="stretch", hide_index=True)
            st.download_button("⬇ Download attendance CSV", register.to_csv(index=False).encode(),
                               "attendance_register.csv", "text/csv")
        with right:
            if vr.preview is not None:
                st.image(vr.preview, caption="Frame with most faces (IDs are anonymous)", width="stretch")
        if len(register) <= 60:
            st.pyplot(charts.attendance_fig(register, threshold))

# ------------------------------------------------------------------ tab 2
with t2:
    st.subheader("Acoustic interaction & noise dynamics")
    st.caption("Audio is split into time windows; spectral energy, zero-crossing rate, spectral flatness and "
               "voice-activity detection classify each window.")
    mic = getattr(st, "audio_input", None)
    opts = ["Upload WAV"] + (["Record from microphone"] if mic else [])
    asrc = st.radio("Audio source", opts, horizontal=True)
    try:
        data = None
        if asrc == "Upload WAV":
            f = st.file_uploader("Classroom audio (WAV)", type=["wav"], key="wav")
            data = f.getvalue() if f else None
        else:
            rec = mic("Record classroom audio")
            data = rec.getvalue() if rec else None
        if data and st.button("Analyze audio", type="primary"):
            with st.spinner("Extracting audio features..."):
                y, sr = au.load_wav(data)
                S.audio = au.analyze_audio(y, sr, source="Uploaded audio")
            st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not process the audio: {exc}")

    if ar is None:
        st.info("Upload a WAV file, record audio, or click **Load demo session** in the sidebar.")
    else:
        aw = ar.windows(float(window_s), float(vad_db))
        mix = (aw["label"].value_counts(normalize=True) * 100).to_dict()
        m = st.columns(4)
        m[0].metric("Duration", fmt_time(ar.duration))
        m[1].metric("Dominant state", aw["label"].mode().iloc[0])
        m[2].metric("Discussion share", f"{mix.get('Student Discussion', 0):.0f}%")
        m[3].metric("Distraction share", f"{mix.get('High Ambient Distraction', 0):.0f}%")
        st.pyplot(charts.audio_fig(aw, float(window_s)))
        c1, c2 = st.columns([2, 3])
        c1.pyplot(charts.mix_fig(mix))
        with c2.expander("Window-level classification table"):
            st.dataframe(aw.drop(columns=["window"]), width="stretch", hide_index=True)

# ------------------------------------------------------------------ tab 3
with t3:
    st.subheader("Multimodal fusion & engagement scoring")
    st.caption("Visual attentiveness (forward head pose, upright posture) and acoustic interaction energy are "
               "merged into one time-step matrix and normalised to an Engagement Index (0-100). "
               "Scores describe the class collectively.")
    if tl.empty or mode == "none":
        st.info("Load visual and/or audio data first (or use the demo session).")
    else:
        st.caption(f"Fusion mode: **{mode}**" + (f" (visual {w_visual}% / audio {100 - w_visual}%)" if mode == "fused" else ""))
        k = st.slider("Replay position (minutes)", 0.0, float(tl["t"].iloc[-1] / 60), float(tl["t"].iloc[-1] / 60),
                      float(window_s) / 60, format="%.1f")
        idx = int(min(len(tl) - 1, round(k * 60 / window_s)))
        S["replay_idx"] = idx
        cur = tl.iloc[idx]
        m = st.columns(4)
        m[0].metric("Engagement now", f"{cur['engagement']:.0f}")
        m[1].metric("Session mean (to here)", f"{tl['engagement'].iloc[:idx + 1].mean():.0f}")
        m[2].metric("Audio state", cur["audio_label"])
        m[3].metric("Faces in window", int(cur["faces"]))
        st.pyplot(charts.engagement_fig(tl, float(low_thr), peaks, dips, idx))
        with st.expander("Time-step fusion matrix"):
            show = tl.copy()
            show["time"] = show["t"].map(fmt_time)
            st.dataframe(show.round(3), width="stretch", hide_index=True)
            st.download_button("⬇ Download matrix CSV", show.to_csv(index=False).encode(),
                               "engagement_timeline.csv", "text/csv")

# ------------------------------------------------------------------ tab 4
with t4:
    st.subheader("AI teaching suggestion engine")
    st.caption("A heuristic advisor watches the collective engagement curve and offers non-intrusive, "
               "class-level prompts. It never singles out individuals.")
    if tl.empty:
        st.info("Load data first (or use the demo session).")
    else:
        only_so_far = st.checkbox("Show only prompts triggered up to the replay position (Tab 3)", False)
        t_limit = (S.get("replay_idx", len(tl) - 1) + 1) * window_s
        shown = [s for s in suggestions if (not only_so_far or s["t"] <= t_limit)]
        if not shown:
            st.success("No prompts needed - engagement looks healthy.")
        for s in shown:
            msg = f"**[{s['time']}] {s['title']}**  \n{s['message']}"
            {"warning": st.warning, "success": st.success}.get(s["severity"], st.info)(msg, icon={
                "warning": "⚠️", "success": "✅"}.get(s["severity"], "💡"))

# ------------------------------------------------------------------ tab 5
with t5:
    st.subheader("Session analytics & educator report")
    if tl.empty and register.empty:
        st.info("Load data first (or use the demo session).")
    else:
        m = st.columns(5)
        m[0].metric("Session length", fmt_time(duration))
        m[1].metric("Attendance", f"{kpis['present']} / {kpis['roster']}" if kpis["roster"] else "n/a")
        m[2].metric("Mean engagement", f"{kpis['mean_eng']:.0f}" if "mean_eng" in kpis else "n/a")
        m[3].metric("Verbal participation",
                    f"{kpis['participation_rate']:.0f}%" if kpis.get("participation_rate") is not None else "n/a")
        m[4].metric("Prompts raised", kpis["prompts"])
        if "mean_eng" in kpis:
            m = st.columns(4)
            m[0].metric("Peak", f"{kpis['peak']:.0f}", fmt_time(kpis["peak_t"]), delta_color="off")
            m[1].metric("Lowest", f"{kpis['low']:.0f}", fmt_time(kpis["low_t"]), delta_color="off")
            m[2].metric("Time engaged (>=60)", f"{kpis['pct_engaged']:.0f}%")
            m[3].metric(f"Time below {low_thr}", f"{kpis['pct_low']:.0f}%")
            st.pyplot(charts.engagement_fig(tl, float(low_thr), peaks, dips))
            if kpis.get("audio_mix"):
                st.pyplot(charts.mix_fig(kpis["audio_mix"]))
        md = report.build_markdown(kpis, register, suggestions, threshold, float(low_thr))
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download session report (.md)", md.encode(), "session_report.md", "text/markdown",
                           type="primary", width="stretch")
        if not register.empty:
            c2.download_button("⬇ Download attendance CSV", register.to_csv(index=False).encode(),
                               "attendance_register.csv", "text/csv", width="stretch")
        with st.expander("Preview report"):
            st.markdown(md)
