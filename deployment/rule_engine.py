"""
rule_engine.py
---------------
ده الملف المسؤول عن "اتخاذ القرار" -- مش بيعمل أي كشف بنفسه، وإنما بياخد:

  - نتائج YOLO لكل شخص (person / helmet / no_helmet / vest / no_vest /
    goggles / no_goggles / fall boxes)
  - نتائج PPEPositionVerifier (من pose_analyzer.py)
  - نتائج FallPoseAnalyzer (من pose_analyzer.py)

وبيحول ده كله لقرارات واضحة: Violations + Fall Status، وبيحتفظ بحالة
(state) لكل عامل عبر الفريمات عشان يقدر يحسب "السقوط استمر قد إيه" ويطلع
تنبيه لو عدى 10 ثواني.

القاعدة الأساسية لـ PPE (helmet/vest/goggles):
    - لو الموديل قال صراحة "no_helmet" / "no_vest" / "no_goggles" ->
      مخالفة مؤكدة (Confirmed) لأن الموديل شاف العامل وقرر إنه من غيرها.
    - لو الموديل معملش detect لأي حاجة خالص (لا الحاجة موجودة ولا
      "no_X") -> برضو نعتبرها مخالفة، لكن بعلامة "Not Detected" عشان
      نفرق بين "شفنا إنه من غيرها" و"مقدرناش نتأكد".
    - لو الحاجة موجودة (helmet/vest/goggles = True) بس الـ pose قال
      "مش في مكانها الصح" -> "Worn Incorrectly".

القاعدة الأساسية للسقوط:
    - YOLO قال Fall + Pose مؤكدها  -> High Confidence Fall
    - واحد بس منهم قال Fall        -> Possible Fall
    - لو الحالة (أي منهم) فضلت مستمرة أكتر من FALL_ALERT_SECONDS ثانية
      من غير انقطاع -> ALERT: "Sustained Fall - Immediate Attention Required"
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import time


FALL_ALERT_SECONDS = 10.0  # المدة اللي لو السقوط استمر عليها يبقى فيه تنبيه


class FallStatus(str, Enum):
    NONE = "none"
    POSSIBLE_FALL = "possible_fall"          # مصدر واحد بس (YOLO أو Pose) قال سقوط
    HIGH_CONFIDENCE_FALL = "high_confidence_fall"  # الاتنين متفقين
    SUSTAINED_FALL_ALERT = "sustained_fall_alert"  # مستمر أكتر من 10 ثواني


@dataclass
class PersonFrameInput:
    """كل المعلومات المطلوبة عن شخص واحد في فريم واحد."""
    person_id: str
    timestamp: float

    # من YOLO -- وجود القطعة
    yolo_helmet_detected: bool = False
    yolo_vest_detected: bool = False
    yolo_goggles_detected: bool = False
    yolo_fall_detected: bool = False

    # من YOLO -- كشف صريح لغياب القطعة (كلاسات no_helmet / no_vest / no_goggles)
    yolo_no_helmet_detected: bool = False
    yolo_no_vest_detected: bool = False
    yolo_no_goggles_detected: bool = False

    # من PPEPositionVerifier.verify(...)
    ppe_position_result: Optional[Dict] = None

    # من FallPoseAnalyzer.analyze(...)
    fall_pose_result: Optional[Dict] = None


@dataclass
class _PersonState:
    """حالة مستمرة لكل شخص بين الفريمات (بنحتفظ بيها جوه الـ RuleEngine)."""
    fall_start_ts: Optional[float] = None
    last_seen_ts: float = 0.0
    already_alerted: bool = False
    # لو الشخص اتغاب عن الفريمات لفترة أطول من كده، نعتبر السقوط انتهى
    # (اتاخد له إسعاف / وقف / حد شاله من الكادر...)
    max_gap_seconds: float = 2.0


@dataclass
class PersonEvaluation:
    """ناتج تقييم شخص واحد -- ده اللي بيتوصل لل Streamlit / التقرير."""
    person_id: str
    timestamp: float
    violations: List[str] = field(default_factory=list)
    fall_status: FallStatus = FallStatus.NONE
    fall_duration_seconds: float = 0.0
    alert: bool = False
    alert_message: Optional[str] = None
    details: Dict = field(default_factory=dict)


class RuleEngine:
    def __init__(self, fall_alert_seconds: float = FALL_ALERT_SECONDS):
        self.fall_alert_seconds = fall_alert_seconds
        self._states: Dict[str, _PersonState] = {}

    # -- PPE ----------------------------------------------------------------

    @staticmethod
    def _evaluate_item(
        item_name: str,
        present_detected: bool,
        absent_detected: bool,
        position_result: Optional[Dict],
    ) -> Optional[str]:
        """
        منطق موحد لأي قطعة PPE (helmet / vest / goggles):
          1) لو الموديل قال صراحة "غير موجودة"  -> مخالفة مؤكدة
          2) لو مفيش أي detection خالص (لا موجودة ولا غير موجودة) -> مخالفة
             لكن بعلامة "مش متأكدين" (ممكن تبقى الرؤية بايظة مش إنها فعلاً غايبة)
          3) لو موجودة لكن في مكان غلط (حسب الـ pose) -> "Worn Incorrectly"
          4) لو موجودة ومكانها صح (أو الـ pose مش واضحة) -> مفيش مخالفة
        """
        if absent_detected:
            return f"Missing {item_name} (Confirmed)"

        if not present_detected:
            return f"Missing {item_name} (Not Detected)"

        if position_result and position_result.get("correct_position") is False:
            return f"{item_name} Worn Incorrectly"

        return None

    def _evaluate_ppe(self, inp: PersonFrameInput) -> List[str]:
        violations: List[str] = []
        ppe = inp.ppe_position_result or {}

        helmet_violation = self._evaluate_item(
            "Helmet", inp.yolo_helmet_detected, inp.yolo_no_helmet_detected, ppe.get("helmet")
        )
        if helmet_violation:
            violations.append(helmet_violation)

        vest_violation = self._evaluate_item(
            "Vest", inp.yolo_vest_detected, inp.yolo_no_vest_detected, ppe.get("vest")
        )
        if vest_violation:
            violations.append(vest_violation)

        goggles_violation = self._evaluate_item(
            "Goggles", inp.yolo_goggles_detected, inp.yolo_no_goggles_detected, ppe.get("goggles")
        )
        if goggles_violation:
            violations.append(goggles_violation)

        return violations

    # -- Fall -----------------------------------------------------------------

    def _get_state(self, person_id: str) -> _PersonState:
        if person_id not in self._states:
            self._states[person_id] = _PersonState()
        return self._states[person_id]

    def _evaluate_fall(self, inp: PersonFrameInput) -> (FallStatus, float, Optional[str]):
        state = self._get_state(inp.person_id)
        pose_says_fall = bool(
            inp.fall_pose_result and inp.fall_pose_result.get("looks_like_fall")
        )
        yolo_says_fall = inp.yolo_fall_detected

        is_falling_now = pose_says_fall or yolo_says_fall

        if not is_falling_now:
            # لو فيه فجوة قصيرة بس (زي فريم اتغلط فيه)، سيبي الحالة شوية قبل
            # ما تصفريها تمامًا -- هنا بنبسطها: أي فريم من غير سقوط يوقف العداد.
            state.fall_start_ts = None
            state.already_alerted = False
            return FallStatus.NONE, 0.0, None

        # فيه سقوط دلوقتي (من مصدر واحد أو الاتنين)
        if state.fall_start_ts is None:
            state.fall_start_ts = inp.timestamp

        duration = inp.timestamp - state.fall_start_ts
        state.last_seen_ts = inp.timestamp

        if pose_says_fall and yolo_says_fall:
            status = FallStatus.HIGH_CONFIDENCE_FALL
        else:
            status = FallStatus.POSSIBLE_FALL

        alert_msg = None
        if duration >= self.fall_alert_seconds:
            status = FallStatus.SUSTAINED_FALL_ALERT
            if not state.already_alerted:
                state.already_alerted = True
            alert_msg = (
                f"ALERT: Worker '{inp.person_id}' has been down for "
                f"{duration:.0f}s (>= {self.fall_alert_seconds:.0f}s) - "
                f"Immediate attention required."
            )

        return status, duration, alert_msg

    # -- Public API -----------------------------------------------------------

    def evaluate(self, inp: PersonFrameInput) -> PersonEvaluation:
        violations = self._evaluate_ppe(inp)
        fall_status, fall_duration, alert_msg = self._evaluate_fall(inp)

        return PersonEvaluation(
            person_id=inp.person_id,
            timestamp=inp.timestamp,
            violations=violations,
            fall_status=fall_status,
            fall_duration_seconds=round(fall_duration, 1),
            alert=fall_status == FallStatus.SUSTAINED_FALL_ALERT,
            alert_message=alert_msg,
            details={
                "ppe_position_result": inp.ppe_position_result,
                "fall_pose_result": inp.fall_pose_result,
            },
        )

    def evaluate_batch(self, inputs: List[PersonFrameInput]) -> List[PersonEvaluation]:
        return [self.evaluate(i) for i in inputs]

    def reset_person(self, person_id: str):
        """لو حد خرج من الكادر نهائيًا وعايزة تصفري حالته يدويًا."""
        self._states.pop(person_id, None)


# ---------------------------------------------------------------------------
# مثال استخدام سريع (مش بيتنفذ إلا لو شغلتي الملف مباشرة)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = RuleEngine(fall_alert_seconds=10.0)

    base_t = time.time()
    # محاكاة: شخص واقع من فريم t=0 لحد t=12 ثانية
    for t_offset in [0, 2, 4, 6, 8, 10, 11, 12]:
        result = engine.evaluate(PersonFrameInput(
            person_id="worker_1",
            timestamp=base_t + t_offset,
            yolo_helmet_detected=True,
            yolo_vest_detected=True,
            yolo_no_vest_detected=False,
            yolo_goggles_detected=False,
            yolo_no_goggles_detected=True,
            yolo_fall_detected=True,
            fall_pose_result={"looks_like_fall": True, "fall_pose_score": 0.8},
        ))
        print(t_offset, result.violations, result.fall_status,
              result.fall_duration_seconds, result.alert, result.alert_message)
