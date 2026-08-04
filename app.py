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
# Fall-detection stabilization
# ----------------------------------------------------------------------
# المشكلة: موديل الفول ديتكشن بيدي False Positive كتير لو اعتمدنا على فريم
# واحد بس، لأن أي وضعية جسم شبه السقوط (انحناء / قرفصة / جلوس على الأرض)
# ممكن تخرج بثقة عالية لفريم واحد أو اتنين بالغلط.
# الحل: مانديش تنبيه "سقوط" إلا لو الكلاس ده فضل يتكرر باستمرار لفترة زمنية
# معينة (مش لحظة واحدة)، ولو اتأكد السقوط، منشيلش التنبيه إلا لو اختفى
# الكلاس فترة كمان (Hysteresis) عشان التنبيه ميرفرفش (يظهر ويختفي بسرعة).

FALL_CLASS_KEYWORD = "fall"  # جزء من اسم الكلاس (مش حساس لحالة الأحرف) - عدّليه لو اسم الكلاس مختلف


class TemporalFallConfirmer:
    """
    بيتابع وجود/غياب كلاس الفول عبر الوقت (مش عدد فريمات ثابت) عشان يفضل
    شغال صح مهما اختلفت سرعة المعالجة (GPU/CPU/جودة الكاميرا).

    confirm_seconds: لازم الفول يفضل ظاهر متواصل للمدة دي عشان يتحول لـ "تأكيد حقيقي"
    clear_seconds:   لازم الفول يختفي متواصل للمدة دي عشان نلغي التأكيد
    """

    def __init__(self, confirm_seconds: float = 1.2, clear_seconds: float = 1.0):
        self.confirm_seconds = confirm_seconds
        self.clear_seconds = clear_seconds
        self._true_since = None
        self._false_since = None
        self.confirmed = False

    def update(self, fall_detected_this_frame: bool, now: float = None) -> bool:
        # في اللايف كاميرا: now = وقت الكمبيوتر الحقيقي (افتراضي).
        # في الفيديو: بنمرر now = frame_idx / fps (وقت الفيديو نفسه)
        # عشان التأكيد يتحسب بالثانية الفعلية جوه الفيديو، مش بسرعة معالجة الفريمات.
        if now is None:
            now = time.time()
        if fall_detected_this_frame:
            self._false_since = None
            if self._true_since is None:
                self._true_since = now
            if not self.confirmed and (now - self._true_since) >= self.confirm_seconds:
                self.confirmed = True
        else:
            self._true_since = None
            if self._false_since is None:
                self._false_since = now
            if self.confirmed and (now - self._false_since) >= self.clear_seconds:
                self.confirmed = False
        return self.confirmed


def frame_has_fall(detections, fall_conf_threshold: float) -> bool:
    """True لو فيه أي Detection اسمه فيه كلمة fall وثقته >= الـ threshold المخصص للفول."""
    return any(
        FALL_CLASS_KEYWORD in d["name"].lower() and d["conf"] >= fall_conf_threshold
        for d in detections
    )


def detections_for_display(detections, fall_conf_threshold: float):
    """
    بيشيل من الرسم (annotate) أي Box فول بثقة أقل من الحد المخصص للفول.
    السبب: البوكس ده أصلاً مش هيتحسب Violation، فمفيش داعي نعرضه ونخوف بيه
    المستخدم من غير لزوم. باقي الكلاسات (خوذة/فيست/جوجلز) بتتعرض زي ما هي.
    """
    return [
        d for d in detections
        if FALL_CLASS_KEYWORD not in d["name"].lower() or d["conf"] >= fall_conf_threshold
    ]


def filter_detections_for_alerts(detections, fall_confirmed: bool):
    """
    بيرجع نسخة من الديتكشنز تتستخدم في حساب التنبيهات/الفيولاشنز بس:
    - لو الفول لسه مش متأكد منه زمنيًا -> بيتشال من القايمة دي (مش من الرسم على الصورة)
    - باقي الفيولاشنز (خوذة/فيست/جوجلز) بتفضل زي ما هي
    """
    filtered = []
    for d in detections:
        if FALL_CLASS_KEYWORD in d["name"].lower():
            if fall_confirmed:
                filtered.append(d)
        else:
            filtered.append(d)
    return filtered

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

    conf_threshold = st.slider("Confidence threshold (PPE)", 0.1, 0.9, 0.4, 0.05)

    st.subheader("🧍 Fall Detection tuning")
    st.caption(
        "الفول ديتكشن أكتر عرضة للـ False Positive، فبناخد ليه ثقة أعلى "
        "وتأكيد زمني قبل ما نطلق إنذار."
    )
    fall_conf_threshold = st.slider("Confidence threshold (Fall)", 0.1, 0.95, 0.6, 0.05)
    confirm_seconds = st.slider("مدة التأكيد قبل الإنذار (ثانية)", 0.2, 3.0, 1.2, 0.1)
    clear_seconds = st.slider("مدة الاختفاء قبل إلغاء الإنذار (ثانية)", 0.2, 3.0, 1.0, 0.1)

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
    """detections هنا المفروض تكون بعد الفلترة (filter_detections_for_alerts)."""
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
            fall_status_slot = col2.empty()
            stop_button = col2.button("Stop")

            fall_confirmer = TemporalFallConfirmer(
                confirm_seconds=confirm_seconds, clear_seconds=clear_seconds
            )

            try:
                while start and not stop_button:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("❌ Lost connection to the camera.")
                        break

                    # نجيب كل الديتكشنز بالـ threshold العام (للـ PPE)
                    detections = detector.predict(frame, conf=conf_threshold)

                    # نحسب هل فيه فول ديتكشن بالـ threshold الخاص بيه (أعلى)
                    fall_now = frame_has_fall(detections, fall_conf_threshold)
                    fall_confirmed = fall_confirmer.update(fall_now)

                    # الرسم على الصورة: بنشيل بوكسات الفول الواطية الثقة عشان منخوفش المستخدم بحاجة مش هتتعتبر Violation أصلاً
                    display_detections = detections_for_display(detections, fall_conf_threshold)
                    annotated = detector.annotate(frame, display_detections)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_slot.image(annotated_rgb, channels="RGB", use_container_width=True)

                    # التنبيه الفعلي بيتحسب من الديتكشنز المفلترة (فول متأكد منه فقط + باقي الفيولاشنز عادي)
                    alert_detections = filter_detections_for_alerts(detections, fall_confirmed)
                    summary = SafetyDetector.violation_summary(alert_detections)

                    if summary:
                        parts = ", ".join(f"{v}x {k}" for k, v in summary.items())
                        alert_slot.error(f"🚨 SAFETY VIOLATION — {parts}")
                    else:
                        alert_slot.success("✅ No violations detected")

                    if fall_now and not fall_confirmed:
                        fall_status_slot.warning("⏳ فول محتمل... بيتم التأكد")
                    else:
                        fall_status_slot.empty()

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
        display_detections = detections_for_display(detections, fall_conf_threshold)
        annotated = detector.annotate(frame, display_detections)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, channels="RGB", use_container_width=True)

        # صورة واحدة = مفيش وقت نتأكد بيه (التأكيد الزمني بيشتغل بس مع فيديو/كاميرا حية).
        # فبنطلب ثقة أعلى (fall_conf_threshold) للفول تحديدًا قبل ما نعتبره Violation فعلي.
        alert_detections = [
            d for d in detections
            if FALL_CLASS_KEYWORD not in d["name"].lower() or d["conf"] >= fall_conf_threshold
        ]
        if any(FALL_CLASS_KEYWORD in d["name"].lower() for d in detections) and not any(
            FALL_CLASS_KEYWORD in d["name"].lower() for d in alert_detections
        ):
            st.info(
                "ℹ️ فيه Detection للفول بس بثقة أقل من الحد المطلوب "
                f"({fall_conf_threshold:.2f}) — متعتبرش Violation، بس شايفينه في الجدول تحت."
            )
        show_violation_banner(alert_detections)

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
            total_violations = {}          # PPE (خوذة/فيست/جوجلز) — لسه بتتعد لكل فريم
            fall_event_count = 0            # عدد "حوادث" السقوط المؤكدة، مش عدد الفريمات
            prev_fall_confirmed = False
            fall_confirmer = TemporalFallConfirmer(
                confirm_seconds=confirm_seconds, clear_seconds=clear_seconds
            )

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                detections = detector.predict(frame, conf=conf_threshold)

                # بنشيل بوكسات الفول الواطية الثقة من الفيديو المرسوم برضه
                display_detections = detections_for_display(detections, fall_conf_threshold)
                annotated = detector.annotate(frame, display_detections)
                writer.write(annotated)

                # بنستخدم وقت الفيديو نفسه (frame_idx / fps) مش وقت المعالجة الفعلي،
                # عشان التأكيد الزمني يبقى صحيح بغض النظر عن سرعة المعالجة.
                video_time = frame_idx / fps
                fall_now = frame_has_fall(detections, fall_conf_threshold)
                fall_confirmed = fall_confirmer.update(fall_now, now=video_time)

                # حادثة سقوط جديدة = اللحظة اللي بيتحول فيها من "مش مؤكد" لـ "مؤكد" بس (مش كل فريم بعد كده)
                if fall_confirmed and not prev_fall_confirmed:
                    fall_event_count += 1
                prev_fall_confirmed = fall_confirmed

                # باقي الـ PPE (خوذة/فيست/جوجلز) لسه بتتعد لكل فريم زي ما هي (مش جزء من مشكلة الفول)
                non_fall_only = [d for d in detections if FALL_CLASS_KEYWORD not in d["name"].lower()]
                for k, v in SafetyDetector.violation_summary(non_fall_only).items():
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

            violation_messages = []
            if fall_event_count > 0:
                # عدد الحوادث (مش الفريمات) — لو الشخص فضل واقع 5 ثواني، ده لسه "حادثة واحدة"
                event_word = "حادثة سقوط" if fall_event_count == 1 else "حوادث سقوط"
                violation_messages.append(f"🚨 {fall_event_count} {event_word} تم تأكيدها (Fall-Detected)")
            if total_violations:
                parts = ", ".join(f"{v}x {k}" for k, v in total_violations.items())
                violation_messages.append(f"🚨 {parts}")

            if violation_messages:
                for msg in violation_messages:
                    st.error(msg)
            else:
                st.success("No violations detected in this video.")

            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="annotated_output.mp4")
    else:
        st.info("Upload an MP4/AVI/MOV video to run detection.")
