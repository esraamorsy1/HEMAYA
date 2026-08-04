import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import pandas as pd
import time

from pose_analyzer import (
    BBox,
    PersonPoseEstimator,
    PPEPositionVerifier,
    FallPoseAnalyzer,
)

from rule_engine import (
    RuleEngine,
    PersonFrameInput,
)

# ===============================
# Model Paths
# ===============================

FALL_MODEL_PATH = "D:\\Hemaya\\best_fall.pt"
PPE_MODEL_PATH = "D:\\Hemaya\\ppe2.pt"
GOGGLES_MODEL_PATH = "D:\\Hemaya\\best_Googles.pt"

# ===============================
# Class Index Map
# ===============================
# ppe_model.names -> {0: 'helmet', 1: 'no helmet', 2: 'no vest', 3: 'person', 4: 'vest'}
# fall_model.names -> confirmed via debug run: 0 = Fall-Detected, 1 = Person
HELMET_CLS = 0
NO_HELMET_CLS = 1
NO_VEST_CLS = 2
VEST_CLS = 4

FALL_DETECTED_CLS = 0
FALL_PERSON_CLS = 1

GOGGLES_CLS = 0
NO_GOGGLES_CLS = 1


# ===============================
# Load Models (cached across reruns)
# ===============================

@st.cache_resource
def load_models():
    fall_model = YOLO(FALL_MODEL_PATH)
    ppe_model = YOLO(PPE_MODEL_PATH)
    goggles_model = YOLO(GOGGLES_MODEL_PATH)
    # A second copy of the same best_fall.pt weights, used only for
    # tracking. We avoid reusing the same model object for two different
    # call styles (.track vs direct call) -- that causes conflicts in the
    # predictor args Ultralytics stores internally on the model object.
    person_tracer = YOLO(FALL_MODEL_PATH)
    return fall_model, ppe_model, goggles_model, person_tracer


@st.cache_resource
def load_pose_pipeline():
    pose_estimator = PersonPoseEstimator()
    ppe_verifier = PPEPositionVerifier()
    fall_pose_analyzer = FallPoseAnalyzer()
    rule_engine = RuleEngine(fall_alert_seconds=10)
    return pose_estimator, ppe_verifier, fall_pose_analyzer, rule_engine


fall_model, ppe_model, goggles_model, person_tracer = load_models()
pose_estimator, ppe_verifier, fall_pose_analyzer, rule_engine = load_pose_pipeline()


# ===============================
# Helpers
# ===============================

def draw_label(frame, x, y, text, color=(0, 0, 255)):
    """
    Draws text with a black background behind it so it stays readable on
    any background, and auto-scales font size to frame width (a 4K video
    needs a much bigger font than a 720p one to stay legible).
    Returns the line height that was drawn (so callers can stack more
    than one line on top of each other).
    """
    frame_w = frame.shape[1]
    font_scale = max(frame_w / 1400.0, 0.6)
    thickness = max(2, int(frame_w / 900))

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y = max(int(y), th + baseline + 10)

    cv2.rectangle(
        frame,
        (int(x), y - th - baseline - 6),
        (int(x) + tw + 10, y + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame, text, (int(x) + 5, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
    )
    return th + baseline + 16


def draw_worker_box(frame, box, worker_id, has_violation, has_alert):
    """
    Always draws the worker's bounding box + ID label, regardless of
    whether they currently have a violation. Box color reflects status:
    green = OK, orange = violation, red = active sustained-fall alert.
    Returns the y position right below the ID label, so violation/alert
    text can be stacked under it.
    """
    x1, y1, x2, y2 = map(int, box)

    if has_alert:
        color = (97, 105, 255)    # light red / salmon
    elif has_violation:
        color = (71, 179, 255)    # light orange
    else:
        color = (144, 238, 144)   # light green

    thickness = max(2, int(frame.shape[1] / 900))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    consumed = draw_label(frame, x1, y1, f"Worker {worker_id}", color=color)
    return y1 + consumed


def _item_status(item_name: str, violations: list) -> str:
    """Turns the rule engine's free-text violations list into a clean
    per-item status for a single PPE piece (helmet/vest/goggles), so the
    dashboard table can show one column per item instead of one long
    combined string."""
    for v in violations:
        if v.startswith(f"Missing {item_name}"):
            return "Missing"
        if v == f"{item_name} Worn Incorrectly":
            return "Incorrect"
    return "Worn"


def build_dashboard_df(evaluations):
    """Current-frame snapshot: one row per worker with a clear Yes/No
    breakdown of what they are and aren't wearing, plus their overall
    violation/fall status."""
    rows = []
    for e in evaluations:
        rows.append({
            "Worker ID": e.person_id,
            "Helmet": _item_status("Helmet", e.violations),
            "Vest": _item_status("Vest", e.violations),
            "Goggles": _item_status("Goggles", e.violations),
            "Has Violation": "YES" if e.violations else "NO",
            "Fall Status": e.fall_status.value,
            "Alert": "YES" if e.alert else "-",
        })
    return pd.DataFrame(rows)


def style_violations(df: pd.DataFrame):
    """Light background colors: soft red for an active alert, soft
    orange for a violation without alert, soft green when everything
    is fine. Text stays dark for readability on light backgrounds."""
    def highlight(row):
        if row["Alert"] == "YES":
            return ["background-color: #ffc2c2; color: #7a0000"] * len(row)
        if row["Has Violation"] == "YES":
            return ["background-color: #ffe8b3; color: #7a4b00"] * len(row)
        return ["background-color: #d9f7d9; color: #12511a"] * len(row)
    return df.style.apply(highlight, axis=1)


def build_worker_log_rows(evaluations, timestamp_str):
    """One structured row per worker per processed frame, meant to
    accumulate into a persistent session log that can later be exported
    and fed into the NLP pipeline."""
    rows = []
    for e in evaluations:
        rows.append({
            "Time": timestamp_str,
            "Worker ID": e.person_id,
            "Helmet": _item_status("Helmet", e.violations),
            "Vest": _item_status("Vest", e.violations),
            "Goggles": _item_status("Goggles", e.violations),
            "Has Violation": "YES" if e.violations else "NO",
            "Violation Details": ", ".join(e.violations) if e.violations else "",
            "Fall Status": e.fall_status.value,
            "Fall Duration (s)": e.fall_duration_seconds,
            "Alert": "YES" if e.alert else "NO",
        })
    return rows


def resize_for_processing(frame, target_width):
    """Downscales the frame to target_width (keeping aspect ratio) before
    it's sent to any model. Returns the frame unchanged if it's already
    narrower than target_width. Since detection and display use the same
    resized frame, there is no need to rescale box coordinates back up."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="HEMAYA", layout="wide", page_icon="🦺")

st.markdown(
    """
    <style>
    .stApp { background-color: #f7f9fc; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e9f0;
        border-radius: 10px;
        padding: 14px 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .hemaya-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1a2b4a;
        margin-bottom: 0;
    }
    .hemaya-subtitle {
        color: #5a6a85;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    </style>
    <div class="hemaya-title">🦺 HEMAYA</div>
    <div class="hemaya-subtitle">Construction Site Safety Dashboard</div>
    """,
    unsafe_allow_html=True,
)

if "alerts_log" not in st.session_state:
    st.session_state.alerts_log = []

if "worker_log" not in st.session_state:
    st.session_state.worker_log = []

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
)

if uploaded_file is not None:

    suffix = uploaded_file.name.split(".")[-1]

    # =====================================
    # IMAGE
    # =====================================
    if suffix.lower() in ["jpg", "jpeg", "png"]:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original Image")

        if st.button("Run Detection"):

            fall_result = fall_model(frame, verbose=False)[0]
            ppe_result = ppe_model(frame, verbose=False)[0]
            goggles_result = goggles_model(frame, verbose=False)[0]

            image = frame.copy()
            image = fall_result.plot(img=image)
            image = ppe_result.plot(img=image)
            image = goggles_result.plot(img=image)

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Detection Result",
            )

    # =====================================
    # VIDEO
    # =====================================
    else:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file.name)

        col_video, col_dashboard = st.columns([2, 1])

        with col_video:
            stframe = st.empty()

        with col_dashboard:
            st.subheader("📊 Live Dashboard")
            metrics_placeholder = st.empty()
            st.markdown("#### ⚠️ Current Worker Status")
            violations_placeholder = st.empty()
            st.markdown("#### 🚨 Fall Alerts Log")
            alerts_placeholder = st.empty()
            st.markdown("#### 📝 Worker Log")
            st.caption(
                "Every processed frame logs each worker's PPE status here. "
                "This accumulates for the whole run and can be downloaded "
                "as CSV below for later use (e.g. the NLP pipeline)."
            )
            worker_log_placeholder = st.empty()
            worker_log_download_placeholder = st.empty()

        st.caption(
            "Note: the Run button processes the entire video in a single run. "
            "If the video is very long, consider trimming it, or increase "
            "'Process every N frames' below for a faster run."
        )

        col_skip, col_size = st.columns(2)
        with col_skip:
            process_every_n = st.slider(
                "Process every N frames (higher = faster, less frequent updates)",
                min_value=1, max_value=10, value=3, step=1,
            )
        with col_size:
            target_width = st.slider(
                "Resize width for processing (lower = faster)",
                min_value=480, max_value=1920, value=960, step=80,
            )

        if st.button("▶ Run Video"):

            last_output = None
            last_evaluations = []
            last_worker_count = 0
            frame_count = 0

            try:
                while cap.isOpened():

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    frame = resize_for_processing(frame, target_width)

                    should_process = (frame_count % process_every_n == 0) or (last_output is None)

                    if should_process:

                        # =============================
                        # Run Models
                        # =============================

                        fall_result = fall_model(frame, verbose=False)[0]

                        fall_detected = False
                        for box in fall_result.boxes:
                            if int(box.cls) == FALL_DETECTED_CLS:
                                fall_detected = True
                                break

                        person_result = person_tracer.track(
                            frame,
                            classes=[FALL_PERSON_CLS],
                            persist=True,
                            tracker="bytetrack.yaml",
                            verbose=False,
                        )[0]

                        ppe_result = ppe_model.predict(frame, conf=0.25, verbose=False)[0]
                        goggles_result = goggles_model.predict(frame, conf=0.5, verbose=False)[0]

                        # =============================
                        # Build Workers Dictionary
                        # =============================

                        workers = {}
                        for box in person_result.boxes:
                            if box.id is None:
                                continue
                            person_id = int(box.id)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            workers[person_id] = {
                                "person_box": (x1, y1, x2, y2),
                                "helmet": False, "no_helmet": False, "helmet_box": None,
                                "vest": False, "no_vest": False, "vest_box": None,
                                "goggles": False, "no_goggles": False, "goggles_box": None,
                                "fall": fall_detected,
                            }

                        # Assign helmet/vest detections to matching person by centroid
                        for box in ppe_result.boxes:
                            cls = int(box.cls)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                            for person_id, person in workers.items():
                                px1, py1, px2, py2 = person["person_box"]
                                if px1 <= cx <= px2 and py1 <= cy <= py2:
                                    if cls == HELMET_CLS:
                                        workers[person_id]["helmet"] = True
                                        workers[person_id]["helmet_box"] = (x1, y1, x2, y2)
                                    elif cls == NO_HELMET_CLS:
                                        workers[person_id]["no_helmet"] = True
                                    elif cls == VEST_CLS:
                                        workers[person_id]["vest"] = True
                                        workers[person_id]["vest_box"] = (x1, y1, x2, y2)
                                    elif cls == NO_VEST_CLS:
                                        workers[person_id]["no_vest"] = True

                        # Assign goggles detections to matching person by centroid
                        for box in goggles_result.boxes:
                            cls = int(box.cls)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                            for person_id, person in workers.items():
                                px1, py1, px2, py2 = person["person_box"]
                                if px1 <= cx <= px2 and py1 <= cy <= py2:
                                    if cls == GOGGLES_CLS:
                                        workers[person_id]["goggles"] = True
                                        workers[person_id]["goggles_box"] = (x1, y1, x2, y2)
                                    elif cls == NO_GOGGLES_CLS:
                                        workers[person_id]["no_goggles"] = True

                        # =============================
                        # MediaPipe + Rule Engine
                        # =============================

                        evaluations = []
                        timestamp = time.time()

                        for worker_id, data in workers.items():

                            person_bbox = BBox(*data["person_box"])
                            pose_output = pose_estimator.estimate(frame, person_bbox)

                            if pose_output is not None:
                                landmarks = pose_output["landmarks_px"]
                                ppe_result_pose = ppe_verifier.verify(
                                    landmarks,
                                    helmet_box=(BBox(*data["helmet_box"]) if data["helmet_box"] else None),
                                    vest_box=(BBox(*data["vest_box"]) if data["vest_box"] else None),
                                    goggles_box=(BBox(*data["goggles_box"]) if data["goggles_box"] else None),
                                )
                                fall_pose_result = fall_pose_analyzer.analyze(landmarks, frame.shape[0])
                            else:
                                ppe_result_pose = None
                                fall_pose_result = None

                            worker_input = PersonFrameInput(
                                person_id=str(worker_id),
                                timestamp=timestamp,
                                yolo_helmet_detected=data["helmet"],
                                yolo_no_helmet_detected=data["no_helmet"],
                                yolo_vest_detected=data["vest"],
                                yolo_no_vest_detected=data["no_vest"],
                                yolo_goggles_detected=data["goggles"],
                                yolo_no_goggles_detected=data["no_goggles"],
                                yolo_fall_detected=data["fall"],
                                ppe_position_result=ppe_result_pose,
                                fall_pose_result=fall_pose_result,
                            )

                            evaluation = rule_engine.evaluate(worker_input)
                            evaluations.append(evaluation)

                            if evaluation.alert and evaluation.alert_message:
                                already_logged = (
                                    st.session_state.alerts_log
                                    and st.session_state.alerts_log[-1]["Message"] == evaluation.alert_message
                                )
                                if not already_logged:
                                    st.session_state.alerts_log.append({
                                        "Time": time.strftime("%H:%M:%S"),
                                        "Worker": evaluation.person_id,
                                        "Message": evaluation.alert_message,
                                    })

                        # Log every worker's status for this processed
                        # frame into the persistent worker log (kept for
                        # the whole run, downloadable as CSV below).
                        st.session_state.worker_log.extend(
                            build_worker_log_rows(evaluations, time.strftime("%H:%M:%S"))
                        )

                        log_df = pd.DataFrame(st.session_state.worker_log)
                        worker_log_placeholder.dataframe(
                            log_df.tail(15)[::-1], use_container_width=True, hide_index=True
                        )
                        worker_log_download_placeholder.download_button(
                            "⬇️ Download Full Worker Log (CSV)",
                            data=log_df.to_csv(index=False).encode("utf-8"),
                            file_name="hemaya_worker_log.csv",
                            mime="text/csv",
                            key=f"worker_log_download_{frame_count}",
                        )

                        # =============================
                        # Draw Model Boxes First (base layer)
                        # =============================
                        output = frame.copy()
                        output = fall_result.plot(img=output)
                        output = ppe_result.plot(img=output)
                        output = goggles_result.plot(img=output)

                        # =============================
                        # Draw Worker Box + ID (always, not just on
                        # violation) + Violations/Alert Text
                        # =============================
                        for e in evaluations:
                            box = workers[int(e.person_id)]["person_box"]
                            y_cursor = draw_worker_box(
                                output, box, e.person_id,
                                has_violation=bool(e.violations),
                                has_alert=e.alert,
                            )

                            if e.violations:
                                text = ", ".join(e.violations)
                                consumed = draw_label(output, box[0], y_cursor, text, color=(71, 179, 255))
                                y_cursor += consumed

                            if e.alert and e.alert_message:
                                draw_label(output, box[0], y_cursor, "ALERT: SUSTAINED FALL", color=(97, 105, 255))

                        last_output = output
                        last_evaluations = evaluations
                        last_worker_count = len(workers)

                    else:
                        # Skipped frame: reuse the last computed overlay
                        # and dashboard data instead of re-running any
                        # model, so the video keeps moving smoothly
                        # without paying the full detection cost on
                        # every single frame.
                        output = last_output
                        evaluations = last_evaluations

                    # =============================
                    # Show Frame
                    # =============================
                    stframe.image(
                        cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                    # =============================
                    # Update Dashboard
                    # =============================
                    total_workers = last_worker_count
                    total_violations = sum(1 for e in evaluations if e.violations)
                    active_alerts = sum(1 for e in evaluations if e.alert)

                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("👷 Workers", total_workers)
                        m2.metric("⚠️ Violations", total_violations)
                        m3.metric("🚨 Active Alerts", active_alerts)

                    df = build_dashboard_df(evaluations)
                    if not df.empty:
                        violations_placeholder.dataframe(
                            style_violations(df), use_container_width=True, hide_index=True
                        )
                    else:
                        violations_placeholder.info("No workers detected in this frame.")

                    if st.session_state.alerts_log:
                        alerts_df = pd.DataFrame(st.session_state.alerts_log[-10:][::-1])
                        alerts_placeholder.dataframe(alerts_df, use_container_width=True, hide_index=True)
                    else:
                        alerts_placeholder.info("No fall alerts yet.")

            finally:
                cap.release()