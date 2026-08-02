"""
HemaAI (حماية) — Worksite Safety Monitor
=========================================
Streamlit app that runs Fall Detection + Helmet/Vest YOLO models.

Modes:
  1. Live Camera  -> real-time detection from the webcam
  2. Upload Image -> run detection on a single photo
  3. Upload Video -> run detection frame-by-frame on a video file, produce
                     an annotated video you can download

If the live camera fails to open (no webcam / permissions / running on a
remote server with no camera), the app shows a clear error and points the
user to the Upload Image / Upload Video modes instead.

Run with:
    streamlit run app.py
"""

import time
import tempfile
import os

import cv2
import numpy as np
import streamlit as st

from detection import SafetyDetector

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------

st.set_page_config(page_title="HemaAI | Safety Monitor", page_icon="🦺", layout="wide")

st.title("🦺 HemaAI — Worksite Safety Monitor")
st.caption("Fall Detection + Helmet/Vest Compliance — YOLO-based")

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Models")
    use_fall = st.checkbox("Fall Detection", value=True)
    use_helmet_vest = st.checkbox("Helmet / Vest Compliance", value=True)
    use_goggles = st.checkbox("Goggles Compliance", value=True)

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

    st.divider()
    st.subheader("Mode")
    mode = st.radio(
        "Choose input source",
        ["📷 Live Camera (real-time)", "🖼️ Upload Image", "🎞️ Upload Video"],
        index=0,
    )

    st.divider()
    st.caption(
        "If the live camera doesn't start (no webcam access, remote server, "
        "browser permissions, etc.), switch to **Upload Image** or "
        "**Upload Video** above — detection works the same way."
    )


@st.cache_resource(show_spinner="Loading models...")
def load_detector(use_fall: bool, use_helmet_vest: bool, use_goggles: bool) -> SafetyDetector:
    return SafetyDetector(use_fall=use_fall, use_helmet_vest=use_helmet_vest, use_goggles=use_goggles)


if not use_fall and not use_helmet_vest and not use_goggles:
    st.warning("Select at least one model from the sidebar to start.")
    st.stop()

detector = load_detector(use_fall, use_helmet_vest, use_goggles)

if detector.errors:
    for err in detector.errors:
        st.warning(f"⚠️ {err}")

if not detector.active_models:
    st.error("No models could be loaded. Please check the `models/` folder — see README.md.")
    st.stop()


def show_violation_banner(detections):
    summary = SafetyDetector.violation_summary(detections)
    if summary:
        parts = ", ".join(f"{v}x {k}" for k, v in summary.items())
        st.error(f"🚨 SAFETY VIOLATION DETECTED — {parts}")
    return summary


# ----------------------------------------------------------------------
# MODE 1 — Live Camera (real-time)
# ----------------------------------------------------------------------

if mode.startswith("📷"):
    st.subheader("Live Camera")

    col1, col2 = st.columns([3, 1])
    with col2:
        camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        start = st.toggle("Start streaming", value=False)
        frame_placeholder_slot = st.empty()

    with col1:
        frame_slot = st.empty()

    if start:
        cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(
                "❌ Couldn't access the camera (index "
                f"{camera_index}). This can happen if there's no webcam, "
                "the app is running on a remote/headless server, or the "
                "camera is being used by another program.\n\n"
                "➡️ Please switch to **Upload Image** or **Upload Video** "
                "from the sidebar instead."
            )
        else:
            alert_slot = st.empty()
            stop_button = col2.button("Stop")

            try:
                while start and not stop_button:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("❌ Lost connection to the camera.")
                        break

                    detections = detector.predict(frame, conf=conf_threshold)
                    annotated = detector.annotate(frame, detections)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    frame_slot.image(annotated_rgb, channels="RGB", use_container_width=True)

                    summary = SafetyDetector.violation_summary(detections)
                    if summary:
                        parts = ", ".join(f"{v}x {k}" for k, v in summary.items())
                        alert_slot.error(f"🚨 SAFETY VIOLATION — {parts}")
                    else:
                        alert_slot.success("✅ No violations detected")

                    time.sleep(0.03)
                    stop_button = col2.button("Stop", key=f"stop_{time.time()}")
            finally:
                cap.release()
    else:
        st.info("Toggle **Start streaming** to begin real-time detection.")

# ----------------------------------------------------------------------
# MODE 2 — Upload Image
# ----------------------------------------------------------------------

elif mode.startswith("🖼️"):
    st.subheader("Upload Image")

    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detections = detector.predict(frame, conf=conf_threshold)
        annotated = detector.annotate(frame, detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, channels="RGB", use_container_width=True)
        show_violation_banner(detections)

        if detections:
            st.subheader("Detections")
            st.table([
                {"Class": d["name"], "Confidence": f'{d["conf"]:.2f}'}
                for d in detections
            ])
        else:
            st.info("No objects detected.")
    else:
        st.info("Upload a JPG/PNG image to run detection.")

# ----------------------------------------------------------------------
# MODE 3 — Upload Video
# ----------------------------------------------------------------------

elif mode.startswith("🎞️"):
    st.subheader("Upload Video")

    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_vid is not None:
        # Save upload to a temp file so OpenCV can read it
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
        tfile_in.write(uploaded_vid.read())
        tfile_in.close()

        cap = cv2.VideoCapture(tfile_in.name)
        if not cap.isOpened():
            st.error("❌ Couldn't read this video file.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tfile_in.name.replace(os.path.splitext(tfile_in.name)[1], "_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Processing video...")
            preview_slot = st.empty()

            frame_idx = 0
            total_violations = {}

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = detector.predict(frame, conf=conf_threshold)
                annotated = detector.annotate(frame, detections)
                writer.write(annotated)

                for k, v in SafetyDetector.violation_summary(detections).items():
                    total_violations[k] = total_violations.get(k, 0) + v

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0),
                                       text=f"Processing frame {frame_idx}/{total_frames}")

                # Show a live-updating preview every few frames
                if frame_idx % 5 == 0:
                    preview_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        channels="RGB", use_container_width=True)

            cap.release()
            writer.release()
            progress.empty()

            st.success("✅ Done processing video.")

            if total_violations:
                parts = ", ".join(f"{v}x {k}" for k, v in total_violations.items())
                st.error(f"🚨 Violations found across the video — {parts}")
            else:
                st.success("No violations detected in this video.")

            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated_output.mp4")
    else:
        st.info("Upload an MP4/AVI/MOV video to run detection.")
