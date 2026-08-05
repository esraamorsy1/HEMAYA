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
MODEL_NAME = "gemini-3.5-flash-lite"

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
    """
    بيدور على الـ API Key في أكتر من مكان، بالترتيب:
      1) متغير بيئة GEMINI_API_KEY
      2) st.secrets (لو الكود شغال جوه Streamlit وقاري .streamlit/secrets.toml بنجاح)
      3) قراءة مباشرة لملف .streamlit/secrets.toml بجانب هذا الملف، احتياطًا
         لو st.secrets مش شغالة لأي سبب (مثلاً الكود اتنفذ برا Streamlit،
         زي test_gemini.py أو FastAPI)

    لو فشل في الثلاثة، بيرمي Error بيوضح بالظبط فين دور وبأي مسار، عشان
    التشخيص يبقى أسهل بدل رسالة عامة.
    """
    checked_locations = []

    api_key = os.environ.get("GEMINI_API_KEY")
    checked_locations.append("متغير البيئة GEMINI_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
            checked_locations.append("st.secrets (عبر Streamlit)")
        except Exception:
            checked_locations.append("st.secrets (مش متاحة -- الكود مش شغال جوه Streamlit؟)")

    secrets_path = None
    if not api_key:
        # احتياطي: قراءة مباشرة لملف .streamlit/secrets.toml بجانب هذا الملف
        # (نفس مجلد nlp_report.py)، من غير الاعتماد على Streamlit خالص.
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            tomllib = None

        secrets_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml"
        )
        checked_locations.append(f"الملف مباشرة: {secrets_path}")

        if tomllib and os.path.isfile(secrets_path):
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            api_key = data.get("GEMINI_API_KEY")

    if not api_key:
        details = "\n  - ".join(checked_locations)
        exists_note = ""
        if secrets_path:
            exists_note = (
                f"\n\nملحوظة: الملف {secrets_path} "
                + ("موجود لكن مفيهوش GEMINI_API_KEY (اتأكدي من اسم المتغير جوّاه)."
                   if os.path.isfile(secrets_path) else "مش موجود في المسار ده أصلاً.")
            )
        raise RuntimeError(
            "مفيش GEMINI_API_KEY في أي مكان من الأماكن دي:\n  - "
            + details
            + exists_note
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

def _build_prompt(
    worker_summaries: list[dict],
    report_date: str | None = None,
    language: str = "ar",
) -> str:
    data_json = json.dumps(worker_summaries, ensure_ascii=False, indent=2)
    report_date = report_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    language_name = "Modern Standard Arabic" if language == "ar" else "English"

    return f"""You are an assistant specialized in writing occupational safety reports
for an industrial worksite, based on data coming from a camera detection system (YOLO)
and a rule engine. The data below is already pre-summarized: each item represents ONE
worker over the whole monitoring period (not a single frame).

Report generation date: {report_date}

Data (JSON):
{data_json}

Meaning of each field:
- worker_id: the worker's identifier
- violations: every distinct PPE violation type (helmet/vest/goggles) recorded for this
  worker during the monitoring period (no duplicates). An empty list means no PPE
  violations were recorded.
- worst_fall_status: the worst fall status this worker reached, meaning exactly:
    * "none": no fall at all
    * "possible_fall": only one source (camera OR pose analysis) suspected a fall
    * "high_confidence_fall": both the camera and pose analysis agreed on a fall
    * "sustained_fall_alert": the worker stayed down for more than 10 continuous
      seconds -- this is an emergency situation
- max_fall_duration_seconds: the longest duration this worker stayed on the ground
- had_alert: whether a real emergency alert actually triggered (true/false)
- frames_logged / frames_evaluated: number of moments this worker was monitored
  (context only, not a severity indicator)
- first_seen / last_seen: first and last date/time this worker was recorded during
  monitoring (format "YYYY-MM-DD HH:MM:SS"), if available

Task: write two professional reports based **strictly and only** on this data —
you must NOT invent or assume any data that is not present in the JSON (such as a
numeric risk score or a prior violation history -- these fields do not exist, do not
reference them). If the data is completely empty, state explicitly that no violations
or incidents were recorded during this period.

Each report must contain:
1. An Executive Summary of 2-3 sentences
2. Details for every worker who has a violation or had_alert=true (worker ID + violation
   type + fall status)
3. Practical safety recommendations, based only on recurring patterns in the data

The report must start with the report date line (Report Date: {report_date}) before
anything else.

Write ONE report, entirely in {language_name}. Output only the report text itself,
with no preamble, no extra commentary, and no other language mixed in."""


# ----------------------------------------------------------------------
# 4) الدالة الرئيسية اللي بتستدعيها من Streamlit
# ----------------------------------------------------------------------

def generate_report(
    data: list[dict],
    report_date: str | None = None,
    language: str = "ar",
) -> dict:
    """
    بتاخد قايمة ديكشنري (ناتج export_for_nlp) وترجع تقرير بلغة واحدة بس
    (مش الاتنين مع بعض -- كده مبنعملش call لـ Gemini من غير داعي للغة
    المستخدم مش هيحملها):

        {"content": "...", "language": "ar" أو "en", "raw": "النص الخام من Gemini"}

    language: "ar" لتقرير عربي، أو "en" لتقرير إنجليزي.
    report_date: تاريخ/وقت توليد التقرير (نص جاهز، مثلاً "2026-08-05 14:23:00").
    لو مش متبعت، بياخد الوقت الحالي تلقائيًا.
    """
    report_date = report_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not data:
        empty_ar = f"Report Date: {report_date}\n\nلا توجد بيانات مخالفات لعمل تقرير عنها في الوقت الحالي."
        empty_en = f"Report Date: {report_date}\n\nNo violation data available to generate a report at this time."
        return {
            "content": empty_ar if language == "ar" else empty_en,
            "language": language,
            "raw": "",
        }

    client = _get_client()
    prompt = _build_prompt(data, report_date=report_date, language=language)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    raw_text = response.text or ""

    return {"content": raw_text.strip(), "language": language, "raw": raw_text}