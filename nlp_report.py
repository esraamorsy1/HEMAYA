"""
nlp_report.py
-------------
الشغلانة الوحيدة بتاعة الملف ده: ياخد نتايج الـ RuleEngine الحقيقية
(person_id, timestamp, violations, fall_status, fall_duration_seconds,
alert, alert_message) — وهي بتيجي فريم فريم لكل عامل — ويعمل منها:

  1. تلخيص لكل عامل (أسوأ حالة وصلها، كل المخالفات المختلفة، هل حصل تنبيه)
  2. تقرير احترافي مكتوب بالعربي والإنجليزي عن طريق Gemini

مفيهوش أي Detection أو موديلات YOLO خالص هنا، ومفيهوش أي بيانات مختلقة —
لو حقل مش موجود في الرول إنجن (زي Risk Score)، مش هيظهر في التقرير.

المتطلبات:
    pip install google-genai

الإعداد (اختاري واحدة بس):
    (أ) متغير بيئة:      set GEMINI_API_KEY=your_key_here   (على Windows CMD)
                          $env:GEMINI_API_KEY="your_key_here" (على PowerShell)
    (ب) ملف Streamlit secrets: حطي في .streamlit/secrets.toml
                          GEMINI_API_KEY = "your_key_here"
"""

import os
import csv
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict

from google import genai

# ممكن تستبدليه بأي موديل تاني متاح عندك (مثلاً "gemini-2.0-flash")
MODEL_NAME = "gemini-2.5-flash"

# ترتيب خطورة حالات السقوط، من الأقل للأكتر -- بنستخدمه عشان نعرف "أسوأ حالة"
# وصلها العامل عبر كل الفريمات اللي اتقيّمت
_FALL_STATUS_SEVERITY = {
    "none": 0,
    "possible_fall": 1,
    "high_confidence_fall": 2,
    "sustained_fall_alert": 3,
}


# ----------------------------------------------------------------------
# 1) الاتصال بـ Gemini
# ----------------------------------------------------------------------

def _get_client() -> genai.Client:
    """بيدور على الـ API Key في أكتر من مكان (متغير بيئة، أو Streamlit secrets)."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "مفيش GEMINI_API_KEY. حطيه كـ متغير بيئة أو في .streamlit/secrets.toml "
            "(شوفي التعليمات فوق الملف)."
        )

    return genai.Client(api_key=api_key)


# ----------------------------------------------------------------------
# 2) "الجسر" بين شكل بيانات الرول إنجن بتاعكم وشكل موحّد نبعته لـ Gemini
# ----------------------------------------------------------------------

def load_rule_engine_csv(csv_path: str) -> list[dict]:
    """
    بتقرا ملف الـ CSV اللي بيطلعه الرول إنجن، وترجعه كقايمة ديكشنري
    (نفس شكل الـ JSON اللي export_for_nlp() مستنياه). كل صف في الـ CSV
    بيتحول لديكشنري واحد، والعمود بيبقى الـ Key.

    استخدام:
        raw_rows = load_rule_engine_csv("rule_engine_output.csv")
        nlp_input = export_for_nlp(raw_rows)
        report = generate_report(nlp_input)
    """
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # كل صف بيتحول لـ dict تلقائي، والعمود الأول بيبقى الـ Key
        return list(reader)


def export_for_nlp(evaluations: list) -> list[dict]:
    """
    بتاخد قايمة نتايج من الرول إنجن (PersonEvaluation objects لو بتاخديها
    مباشرة من الذاكرة، أو dicts لو جايالك من CSV) وترجعها كقايمة ديكشنري
    موحّدة الشكل، بنفس أسامي الحقول الحقيقية اللي الرول إنجن بيطلعها:
    person_id, timestamp, violations, fall_status, fall_duration_seconds,
    alert, alert_message.

    مفيش أي حقل مختلق هنا (زي Risk Score) -- لو مش موجود في الرول إنجن
    مش هيظهر خالص.
    """
    formatted = []
    for record in evaluations:
        # لو جالك PersonEvaluation object مباشرة من الذاكرة (dataclass)
        if is_dataclass(record) and not isinstance(record, dict):
            row = asdict(record)
        else:
            row = dict(record)

        # الـ fall_status ممكن يكون Enum (FallStatus.NONE) أو نص عادي (لو جاي من CSV)
        fall_status = row.get("fall_status")
        fall_status = getattr(fall_status, "value", fall_status)

        # الـ violations ممكن تيجي كقايمة حقيقية (من الذاكرة)، أو كنص لو جاية من CSV
        # (زي "['Missing Helmet (Confirmed)']") -- بنتعامل مع الحالتين
        violations = row.get("violations", [])
        if isinstance(violations, str):
            violations = [v.strip(" '\"") for v in violations.strip("[]").split(",") if v.strip()]

        # alert ممكن تيجي كنص "True"/"False" لو جاية من CSV
        alert = row.get("alert", False)
        if isinstance(alert, str):
            alert = alert.strip().lower() == "true"

        formatted.append({
            "person_id": row.get("person_id"),
            "timestamp": row.get("timestamp"),
            "violations": violations,
            "fall_status": fall_status or "none",
            "fall_duration_seconds": float(row.get("fall_duration_seconds") or 0),
            "alert": alert,
            "alert_message": row.get("alert_message"),
        })
    return formatted


def summarize_worker_log(worker_log_rows: list[dict]) -> list[dict]:
    """
    مخصوصة لشكل st.session_state.worker_log بالظبط (اللي بتبنيه دالة
    build_worker_log_rows في app.py): كل صف فيه Time, Worker ID, Helmet,
    Vest, Goggles, Has Violation, Violation Details, Fall Status,
    Fall Duration (s), Alert.

    بتلخّص كل عامل (Worker ID) في سطر واحد بس، بدل ما يتكرر نفس الكلام
    لكل فريم اتسجل فيه.
    """
    by_worker: Dict[str, dict] = defaultdict(lambda: {
        "worker_id": None,
        "distinct_violations": set(),
        "worst_fall_status": "none",
        "max_fall_duration_seconds": 0.0,
        "had_alert": False,
        "frames_logged": 0,
        "first_seen": None,
        "last_seen": None,
    })

    for row in worker_log_rows:
        worker_id = row.get("Worker ID")
        w = by_worker[worker_id]
        w["worker_id"] = worker_id
        w["frames_logged"] += 1

        details = row.get("Violation Details") or ""
        if details:
            w["distinct_violations"].update(v.strip() for v in details.split(",") if v.strip())

        fall_status = row.get("Fall Status", "none")
        if _FALL_STATUS_SEVERITY.get(fall_status, 0) > _FALL_STATUS_SEVERITY.get(w["worst_fall_status"], 0):
            w["worst_fall_status"] = fall_status

        try:
            duration = float(row.get("Fall Duration (s)") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        w["max_fall_duration_seconds"] = max(w["max_fall_duration_seconds"], duration)

        if str(row.get("Alert", "")).strip().upper() == "YES":
            w["had_alert"] = True

        t = row.get("Time")
        d = row.get("Date")
        stamp = f"{d} {t}" if d and t else (t or d)
        if stamp:
            w["first_seen"] = w["first_seen"] or stamp
            w["last_seen"] = stamp

    return [
        {
            "worker_id": w["worker_id"],
            "violations": sorted(w["distinct_violations"]),
            "worst_fall_status": w["worst_fall_status"],
            "max_fall_duration_seconds": round(w["max_fall_duration_seconds"], 1),
            "had_alert": w["had_alert"],
            "frames_logged": w["frames_logged"],
            "first_seen": w["first_seen"],
            "last_seen": w["last_seen"],
        }
        for w in by_worker.values()
    ]


def summarize_by_worker(records: list[dict]) -> list[dict]:
    """
    الرول إنجن بيطلع نتيجة لكل عامل في كل فريم -- يعني نفس العامل ممكن يظهر
    مئات المرات. الدالة دي بتلخّص كل عامل في سطر واحد بس قبل ما نبعت لـ
    Gemini، عشان التقرير يبقى مفيد ومختصر بدل ما يتكرر نفس الكلام مرات كتير.

    لكل عامل بترجع:
      - كل المخالفات المختلفة اللي ظهرت له (من غير تكرار)
      - أسوأ fall_status وصله
      - أطول مدة سقوط مسجلة
      - هل حصل تنبيه (alert) في أي لحظة
      - عدد الفريمات اللي اتقيّم فيها (سياق بس، مش للتقرير بالضرورة)
    """
    by_worker: Dict[str, dict] = defaultdict(lambda: {
        "person_id": None,
        "distinct_violations": set(),
        "worst_fall_status": "none",
        "max_fall_duration_seconds": 0.0,
        "had_alert": False,
        "alert_message": None,
        "frames_evaluated": 0,
    })

    for r in records:
        w = by_worker[r["person_id"]]
        w["person_id"] = r["person_id"]
        w["distinct_violations"].update(r["violations"])
        w["frames_evaluated"] += 1
        w["max_fall_duration_seconds"] = max(w["max_fall_duration_seconds"], r["fall_duration_seconds"])

        if _FALL_STATUS_SEVERITY.get(r["fall_status"], 0) > _FALL_STATUS_SEVERITY.get(w["worst_fall_status"], 0):
            w["worst_fall_status"] = r["fall_status"]

        if r["alert"]:
            w["had_alert"] = True
            w["alert_message"] = r["alert_message"]

    return [
        {
            "person_id": w["person_id"],
            "violations": sorted(w["distinct_violations"]),
            "worst_fall_status": w["worst_fall_status"],
            "max_fall_duration_seconds": round(w["max_fall_duration_seconds"], 1),
            "had_alert": w["had_alert"],
            "alert_message": w["alert_message"],
            "frames_evaluated": w["frames_evaluated"],
        }
        for w in by_worker.values()
    ]


# ----------------------------------------------------------------------
# 3) بناء الـ Prompt اللي هيتبعت لـ Gemini
# ----------------------------------------------------------------------

def _build_prompt(worker_summaries: list[dict], report_date: str | None = None) -> str:
    data_json = json.dumps(worker_summaries, ensure_ascii=False, indent=2)
    report_date = report_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""أنت مساعد متخصص في كتابة تقارير السلامة المهنية (Occupational Safety Reports)
لموقع عمل صناعي، بناءً على بيانات جاية من نظام كشف بالكاميرا (YOLO) ومحرك قواعد (Rule Engine).
البيانات دي ملخّصة مسبقًا: كل عنصر يمثل عامل واحد على مدار فترة المراقبة (مش فريم واحد).

تاريخ إصدار هذا التقرير: {report_date}
Report generation date: {report_date}

البيانات (JSON):
{data_json}

معنى كل حقل:
- worker_id: معرف العامل
- violations: كل أنواع مخالفات معدات الحماية (خوذة/فيست/نظارة) اللي اتسجلت للعامل ده
  خلال فترة المراقبة (من غير تكرار). لو القايمة فاضية، معناها مفيش مخالفات معدات.
- worst_fall_status: أسوأ حالة سقوط وصلها العامل، وده معناها بالظبط:
    * "none": مفيش سقوط خالص
    * "possible_fall": مصدر واحد بس (الكاميرا أو تحليل الوضعية) اشتبه في سقوط
    * "high_confidence_fall": الكاميرا وتحليل الوضعية اتفقوا الاتنين على سقوط
    * "sustained_fall_alert": العامل فضل واقع أكتر من 10 ثواني متواصلة -- ده وضع طارئ
- max_fall_duration_seconds: أطول مدة فضل فيها العامل واقع على الأرض
- had_alert: هل حصل تنبيه طارئ فعلي (true/false)
- frames_logged / frames_evaluated: عدد اللحظات اللي اتراقب فيها العامل (سياق بس، مش مؤشر خطورة)
- first_seen / last_seen: أول وآخر تاريخ ووقت اتسجل فيه العامل في المراقبة (بالصيغة "YYYY-MM-DD HH:MM:SS")، لو متاحة

المطلوب: اكتب تقريرين احترافيين مبنيين على البيانات دي **بالظبط فقط** —
ممنوع تختلقي أو تفترضي أي بيانات مش موجودة في الـ JSON (زي درجة خطورة رقمية
أو تاريخ مخالفات سابق -- الحقول دي مش موجودة، متذكريهاش خالص). لو القايمة
فاضية بالكامل، قولي صراحة إن مفيش أي مخالفات أو حوادث مسجلة في الفترة دي.

كل تقرير لازم يحتوي على:
1. ملخص تنفيذي (Executive Summary) من 2-3 جمل
2. تفاصيل كل عامل عنده مخالفة أو had_alert=true (رقمه + نوع المخالفة + حالة السقوط)
3. توصيات عملية لتحسين السلامة، مبنية على الأنماط المتكررة في البيانات فقط

كل تقرير لازم يبدأ بسطر تاريخ الإصدار (Report Date: {report_date}) قبل أي حاجة تانية.

اكتبي الناتج **بالظبط** بالتنسيق ده، من غير أي مقدمة أو كلام زيادة قبل أو بعد:

===ARABIC===
(التقرير بالعربي هنا)
===ENGLISH===
(English report here)"""


# ----------------------------------------------------------------------
# 4) الدالة الرئيسية اللي بتستدعيها من Streamlit
# ----------------------------------------------------------------------

def generate_report(data: list[dict], report_date: str | None = None) -> dict:
    """
    بتاخد قايمة ديكشنري (ناتج export_for_nlp) وترجع:
        {"ar": "...", "en": "...", "raw": "النص الخام كامل من Gemini"}

    report_date: تاريخ/وقت توليد التقرير (نص جاهز، مثلاً "2026-08-05 14:23:00").
    لو مش متبعت، بياخد الوقت الحالي تلقائيًا.
    """
    report_date = report_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not data:
        return {
            "ar": f"Report Date: {report_date}\n\nلا توجد بيانات مخالفات لعمل تقرير عنها في الوقت الحالي.",
            "en": f"Report Date: {report_date}\n\nNo violation data available to generate a report at this time.",
            "raw": "",
        }

    client = _get_client()
    prompt = _build_prompt(data, report_date=report_date)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    raw_text = response.text or ""

    if "===ARABIC===" in raw_text and "===ENGLISH===" in raw_text:
        ar_report = raw_text.split("===ARABIC===")[1].split("===ENGLISH===")[0].strip()
        en_report = raw_text.split("===ENGLISH===")[1].strip()
    else:
        # لو Gemini مطبقش الفورمات المطلوب بالظبط، نرجع نفس النص في الاتنين
        # بدل ما نطلع Error، عشان التقرير يفضل متاح حتى لو الفورمات مش مضبوط 100%
        ar_report = raw_text
        en_report = raw_text

    return {"ar": ar_report, "en": en_report, "raw": raw_text}
