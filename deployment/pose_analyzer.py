"""
pose_analyzer.py
-----------------
طبقة تحليل إضافية فوق YOLO باستخدام MediaPipe Pose.
مسؤول عن حاجتين بس (زي ما اتفقنا):

1) PPEPositionVerifier  -> يتأكد إن الـ Helmet / Vest / Goggles فعلاً في مكانها
الصح على الجسم (مش بس "موجودة في الصورة" زي ما YOLO بيقول).

2) FallPoseAnalyzer      -> يحسب زاوية الجذع + وضع الجسم (أفقي/رأسي) + قرب
الراس من الأرض، عشان يدي إشارة "الوضعية شكلها سقوط" تتبني عليها منطق
الزمن في rule_engine.py.

الملف ده معتمد على mediapipe + numpy بس، ومش بيعرف حاجة عن الـ Rule Engine
أو التقرير النهائي -- ده شغل ملف تاني (rule_engine.py) عن قصد، عشان الفصل
بين "استخراج الإشارات" و"اتخاذ القرار".
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import math
import threading

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# أسماء اللاندماركس المهمة لينا (مؤشرات MediaPipe الرسمية)
NOSE = mp_pose.PoseLandmark.NOSE
LEFT_EYE = mp_pose.PoseLandmark.LEFT_EYE
RIGHT_EYE = mp_pose.PoseLandmark.RIGHT_EYE
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE


@dataclass
class BBox:
    """بوكس بسيط: (x1, y1, x2, y2) بالإحداثيات الحقيقية على الفريم الكامل."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


class PersonPoseEstimator:
    """
    Wrapper حول mediapipe.Pose بيشتغل على crop خاص بكل شخص (جاي من YOLO
    person bbox)، وبيرجع اللاندماركس بالإحداثيات الحقيقية على الفريم
    الأصلي (مش على الـ crop) عشان نقدر نقارنها ببوكسات الـ PPE.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        # static_image_mode=True لأن estimate() بتتنادى أكتر من مرة في نفس
        # الفريم -- مرة لكل عامل مختلف (crop مختلف تمامًا كل مرة) -- ده مش
        # فيديو مستمر لنفس الشخص، فمينفعش نستخدم وضع الـ tracking الداخلي
        # بتاع MediaPipe (static_image_mode=False) لأنه بيعتمد على timestamps
        # متزايدة بشكل صارم لنفس الـ subject، وده كان بيسبب:
        # "Packet timestamp mismatch ... Current minimum expected timestamp"
        self._pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # الـ instance ده بيتشارك (عن طريق st.cache_resource) بين كل
        # الـ sessions/threads بتاعت الـ Streamlit app. mediapipe.Pose
        # مش thread-safe -- لو تريدين مختلفين نادوا process() في نفس
        # الوقت، الـ internal timestamp counter بتاعه بيتلخبط وبيطلع
        # "Packet timestamp mismatch". الـ Lock ده بيضمن إن نداء واحد بس
        # يدخل الـ graph في كل لحظة.
        self._lock = threading.Lock()

    def estimate(self, frame: np.ndarray, person_bbox: BBox,
                 margin_ratio: float = 0.15) -> Optional[Dict]:
        """
        frame: الفريم الكامل (BGR, زي ما بييجي من OpenCV)
        person_bbox: بوكس الشخص من YOLO (إحداثيات حقيقية)
        margin_ratio: هامش حوالين البوكس عشان الأطراف متتقصش

        بيرجع dict فيه:
            landmarks_px: dict {landmark_id: (x, y, visibility)} بإحداثيات الفريم الأصلي
            crop_box: BBox المستخدم فعليًا (بعد الهامش)
        أو None لو معرفش يكشف حد في الكروب ده.
        """
        h_frame, w_frame = frame.shape[:2]

        mx = person_bbox.w * margin_ratio
        my = person_bbox.h * margin_ratio
        x1 = max(0, int(person_bbox.x1 - mx))
        y1 = max(0, int(person_bbox.y1 - my))
        x2 = min(w_frame, int(person_bbox.x2 + mx))
        y2 = min(h_frame, int(person_bbox.y2 + my))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        rgb_crop = crop[:, :, ::-1]  # BGR -> RGB
        with self._lock:
            result = self._pose.process(rgb_crop)

        if not result.pose_landmarks:
            return None

        crop_h, crop_w = crop.shape[:2]
        landmarks_px = {}
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            # تحويل من نسبي (0-1) داخل الكروب -> بكسل حقيقي على الفريم الأصلي
            px = x1 + lm.x * crop_w
            py = y1 + lm.y * crop_h
            landmarks_px[idx] = (px, py, lm.visibility)

        return {
            "landmarks_px": landmarks_px,
            "crop_box": BBox(x1, y1, x2, y2),
        }

    def close(self):
        self._pose.close()


# ---------------------------------------------------------------------------
# 1) PPE Position Verification
# ---------------------------------------------------------------------------

class PPEPositionVerifier:
    """
    بياخد لاندماركس الجسم + بوكسات الـ PPE اللي طلعتلها YOLO، وبيقرر هل
    كل قطعة في مكانها الصح فعلاً ولا لأ.

    الفلسفة: YOLO بيقول "فيه هيلمت في الصورة"، إحنا بنتأكد إنه "فوق الراس"
    مش حاجة تانية زي كاب عادي أو حاجة واقعة جنب حد.
    """

    def __init__(self,
                 helmet_max_dist_ratio: float = 0.6,
                 goggles_max_dist_ratio: float = 0.5,
                 min_visibility: float = 0.4):
        # الـ ratio بيتحسب بالنسبة لطول الجذع (كتف->حوض) عشان يبقى
        # مستقل عن حجم الشخص في الفريم (قريب من الكاميرا / بعيد)
        self.helmet_max_dist_ratio = helmet_max_dist_ratio
        self.goggles_max_dist_ratio = goggles_max_dist_ratio
        self.min_visibility = min_visibility

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _bbox_center(b: Optional[BBox]) -> Optional[Tuple[float, float]]:
        return (b.cx, b.cy) if b else None

    def _torso_length(self, lm: Dict) -> Optional[float]:
        """طول الجذع كوحدة قياس نسبية (كتفين -> حوض)."""
        try:
            sh_mid = (
                (lm[LEFT_SHOULDER][0] + lm[RIGHT_SHOULDER][0]) / 2,
                (lm[LEFT_SHOULDER][1] + lm[RIGHT_SHOULDER][1]) / 2,
            )
            hip_mid = (
                (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2,
                (lm[LEFT_HIP][1] + lm[RIGHT_HIP][1]) / 2,
            )
            length = self._dist(sh_mid, hip_mid)
            return length if length > 1e-3 else None
        except KeyError:
            return None

    def check_helmet(self, lm: Dict, helmet_box: Optional[BBox]) -> Dict:
        if helmet_box is None:
            return {"present": False, "correct_position": False, "reason": "helmet_not_detected"}

        torso = self._torso_length(lm)
        if torso is None:
            return {"present": True, "correct_position": None, "reason": "pose_unclear"}

        nose = lm.get(NOSE)
        if nose is None or nose[2] < self.min_visibility:
            return {"present": True, "correct_position": None, "reason": "head_not_visible"}

        head_point = (nose[0], nose[1])
        helmet_center = self._bbox_center(helmet_box)
        dist = self._dist(head_point, helmet_center)

        # الخوذة المفروض فوق الراس شوية، مش بعيدة عنه
        max_dist = torso * self.helmet_max_dist_ratio
        is_above = helmet_center[1] <= head_point[1] + (torso * 0.15)  # فوق الراس تقريبًا
        correct = dist <= max_dist and is_above

        return {
            "present": True,
            "correct_position": correct,
            "reason": "ok" if correct else "helmet_misaligned",
        }

    def check_vest(self, lm: Dict, vest_box: Optional[BBox]) -> Dict:
        if vest_box is None:
            return {"present": False, "correct_position": False, "reason": "vest_not_detected"}

        try:
            sh_l, sh_r = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
            hip_l, hip_r = lm[LEFT_HIP], lm[RIGHT_HIP]
        except KeyError:
            return {"present": True, "correct_position": None, "reason": "pose_unclear"}

        if min(sh_l[2], sh_r[2], hip_l[2], hip_r[2]) < self.min_visibility:
            return {"present": True, "correct_position": None, "reason": "torso_not_visible"}

        torso_x1 = min(sh_l[0], sh_r[0], hip_l[0], hip_r[0])
        torso_x2 = max(sh_l[0], sh_r[0], hip_l[0], hip_r[0])
        torso_y1 = min(sh_l[1], sh_r[1])
        torso_y2 = max(hip_l[1], hip_r[1])

        vest_center = self._bbox_center(vest_box)

        # لازم مركز الفيست يقع تقريبًا جوه منطقة الجذع (كتفين -> حوض)
        pad_x = (torso_x2 - torso_x1) * 0.3
        pad_y = (torso_y2 - torso_y1) * 0.3
        correct = (
            (torso_x1 - pad_x) <= vest_center[0] <= (torso_x2 + pad_x)
            and (torso_y1 - pad_y) <= vest_center[1] <= (torso_y2 + pad_y)
        )

        return {
            "present": True,
            "correct_position": correct,
            "reason": "ok" if correct else "vest_misaligned",
        }

    def check_goggles(self, lm: Dict, goggles_box: Optional[BBox]) -> Dict:
        if goggles_box is None:
            return {"present": False, "correct_position": False, "reason": "goggles_not_detected"}

        torso = self._torso_length(lm)
        try:
            eye_l, eye_r = lm[LEFT_EYE], lm[RIGHT_EYE]
        except KeyError:
            return {"present": True, "correct_position": None, "reason": "pose_unclear"}

        if torso is None or min(eye_l[2], eye_r[2]) < self.min_visibility:
            return {"present": True, "correct_position": None, "reason": "eyes_not_visible"}

        eyes_mid = ((eye_l[0] + eye_r[0]) / 2, (eye_l[1] + eye_r[1]) / 2)
        goggles_center = self._bbox_center(goggles_box)
        dist = self._dist(eyes_mid, goggles_center)

        max_dist = torso * self.goggles_max_dist_ratio
        correct = dist <= max_dist

        return {
            "present": True,
            "correct_position": correct,
            "reason": "ok" if correct else "goggles_misaligned",
        }

    def verify(self, lm: Dict, helmet_box: Optional[BBox] = None,
            vest_box: Optional[BBox] = None,
            goggles_box: Optional[BBox] = None) -> Dict:
        """نتيجة شاملة للثلاثة قطع مرة واحدة."""
        return {
            "helmet": self.check_helmet(lm, helmet_box),
            "vest": self.check_vest(lm, vest_box),
            "goggles": self.check_goggles(lm, goggles_box),
        }


# ---------------------------------------------------------------------------
# 2) Fall Pose Analysis (زاوية الجذع + أفقية الجسم + قرب الراس من الأرض)
# ---------------------------------------------------------------------------

class FallPoseAnalyzer:
    """
    بيحسب من اللاندماركس بس (من غير أي منطق زمن -- ده شغل rule_engine):
      - trunk_angle_deg: زاوية الجذع بالنسبة للرأسي (0 = واقف تمامًا، 90 = مستلقي أفقي)
      - is_horizontal: هل الجسم أقرب للوضع الأفقي
      - head_near_ground: هل الراس قريبة نسبيًا من أسفل الفريم
      - knees_bent: هل الركب مطوية (مؤشر إضافي)
      - fall_pose_score: 0..1 (كل ما زاد يبقى شكل الوضعية أقرب للسقوط)
    """

    def __init__(self,
                 horizontal_angle_threshold_deg: float = 55.0,
                 head_ground_ratio_threshold: float = 0.85,
                 min_visibility: float = 0.4):
        self.horizontal_angle_threshold_deg = horizontal_angle_threshold_deg
        self.head_ground_ratio_threshold = head_ground_ratio_threshold
        self.min_visibility = min_visibility

    @staticmethod
    def _angle_from_vertical(p_top: Tuple[float, float],
                              p_bottom: Tuple[float, float]) -> float:
        dx = p_bottom[0] - p_top[0]
        dy = p_bottom[1] - p_top[1]
        # زاوية الخط بالنسبة للمحور الرأسي (y)
        angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
        return math.degrees(angle_rad)

    def analyze(self, lm: Dict, frame_height: int) -> Optional[Dict]:
        try:
            sh_l, sh_r = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
            hip_l, hip_r = lm[LEFT_HIP], lm[RIGHT_HIP]
            nose = lm[NOSE]
        except KeyError:
            return None

        vis_ok = min(sh_l[2], sh_r[2], hip_l[2], hip_r[2]) >= self.min_visibility
        if not vis_ok:
            return None

        sh_mid = ((sh_l[0] + sh_r[0]) / 2, (sh_l[1] + sh_r[1]) / 2)
        hip_mid = ((hip_l[0] + hip_r[0]) / 2, (hip_l[1] + hip_r[1]) / 2)

        trunk_angle = self._angle_from_vertical(sh_mid, hip_mid)
        is_horizontal = trunk_angle >= self.horizontal_angle_threshold_deg

        head_ratio = nose[1] / max(frame_height, 1)  # 0 فوق، 1 تحت
        head_near_ground = head_ratio >= self.head_ground_ratio_threshold

        knees_bent = False
        try:
            knee_l, knee_r = lm[LEFT_KNEE], lm[RIGHT_KNEE]
            ankle_l, ankle_r = lm[LEFT_ANKLE], lm[RIGHT_ANKLE]
            if min(knee_l[2], knee_r[2], ankle_l[2], ankle_r[2]) >= self.min_visibility:
                # لو الركبة قريبة عموديًا من الحوض والكاحل (مش ممدودة) -> مطوية
                knee_hip_dy = abs(hip_mid[1] - ((knee_l[1] + knee_r[1]) / 2))
                knee_ankle_dy = abs(((knee_l[1] + knee_r[1]) / 2) - ((ankle_l[1] + ankle_r[1]) / 2))
                knees_bent = knee_ankle_dy < knee_hip_dy * 0.6
        except KeyError:
            pass

        # سكور بسيط: وزن أكبر للأفقية وقرب الراس من الأرض
        score = 0.0
        score += 0.5 if is_horizontal else (trunk_angle / self.horizontal_angle_threshold_deg) * 0.3
        score += 0.35 if head_near_ground else 0.0
        score += 0.15 if knees_bent else 0.0
        score = min(score, 1.0)

        return {
            "trunk_angle_deg": round(trunk_angle, 1),
            "is_horizontal": is_horizontal,
            "head_near_ground": head_near_ground,
            "knees_bent": knees_bent,
            "fall_pose_score": round(score, 2),
            # عتبة عملية: لو السكور أعلى من 0.55 نعتبرها "وضعية سقوط محتملة"
            "looks_like_fall": score >= 0.55,
        }