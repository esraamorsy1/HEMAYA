"""
detection.py
------------
Core logic for loading the two YOLO models (Fall Detection + Helmet/Vest)
and running inference on a single frame, merging results into one
unified list of detections.

Classes are grouped into:
  - VIOLATION classes -> drawn in red, trigger an alert
  - SAFE / neutral classes -> drawn in green

Fall Detection model classes : ['Fall-Detected']
Helmet/Vest model classes    : ['helmet', 'no helmet', 'no vest', 'person', 'vest']
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FALL_MODEL_PATH = os.path.join(MODELS_DIR, "fall_detection_best.pt")
HELMET_VEST_MODEL_PATH = os.path.join(MODELS_DIR, "helmet_vest_best.pt")
GOGGLES_MODEL_PATH = os.path.join(MODELS_DIR, "goggles_best.pt")
# Classes that represent a safety VIOLATION (drawn red + counted in alerts)
VIOLATION_CLASSES = {"no helmet", "no vest", "fall-detected", "no goggles"}

# Fixed colors (BGR, for OpenCV)
COLOR_VIOLATION = (0, 0, 255)      # red
COLOR_SAFE = (0, 200, 0)           # green
COLOR_NEUTRAL = (255, 180, 0)      # blue-ish, for 'person'


def _color_for_class(class_name: str):
    name = class_name.lower()
    if name in VIOLATION_CLASSES:
        return COLOR_VIOLATION
    if name == "person":
        return COLOR_NEUTRAL
    return COLOR_SAFE


class SafetyDetector:
    """Wraps one or two YOLO models and runs combined inference."""

    def __init__(self, use_fall=True, use_helmet_vest=True, use_goggles=True):
        self.fall_model = None
        self.helmet_vest_model = None
        self.goggles_model = None
        self.errors = []

        if use_fall:
            if os.path.exists(FALL_MODEL_PATH):
                self.fall_model = YOLO(FALL_MODEL_PATH)
            else:
                self.errors.append(f"Fall Detection weights not found at {FALL_MODEL_PATH}")

        if use_helmet_vest:
            if os.path.exists(HELMET_VEST_MODEL_PATH):
                self.helmet_vest_model = YOLO(HELMET_VEST_MODEL_PATH)
            else:
                self.errors.append(
                    f"Helmet/Vest weights not found at {HELMET_VEST_MODEL_PATH}. "
                    f"Place your best.pt there (see README)."
                )

        if use_goggles:
            if os.path.exists(GOGGLES_MODEL_PATH):
                self.goggles_model = YOLO(GOGGLES_MODEL_PATH)
            else:
                self.errors.append(
                    f"Goggles weights not found at {GOGGLES_MODEL_PATH}. "
                    f"Place your best.pt there (see README)."
                )

    @property
    def active_models(self):
        return [m for m in (self.fall_model, self.helmet_vest_model, self.goggles_model) if m is not None]

    def predict(self, frame_bgr: np.ndarray, conf: float = 0.4):
        """
        Runs every loaded model on the frame and returns a merged list of
        detections: [{'name', 'conf', 'xyxy', 'is_violation'}, ...]
        """
        detections = []

        for model in self.active_models:
            results = model.predict(source=frame_bgr, conf=conf, verbose=False)
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "name": name,
                    "conf": confidence,
                    "xyxy": xyxy,
                    "is_violation": name.lower() in VIOLATION_CLASSES,
                })

        return detections

    def annotate(self, frame_bgr: np.ndarray, detections):
        """Draws boxes + labels on a copy of the frame. Returns annotated frame."""
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
            color = _color_for_class(det["name"])
            label = f'{det["name"]} {det["conf"]:.2f}'

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    @staticmethod
    def violation_summary(detections):
        """Returns a dict {class_name: count} for violation classes only."""
        summary = {}
        for det in detections:
            if det["is_violation"]:
                summary[det["name"]] = summary.get(det["name"], 0) + 1
        return summary
