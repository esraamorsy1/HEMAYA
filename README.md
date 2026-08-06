# HEMAYA — Construction Site Safety Dashboard

Streamlit app that watches a construction site (image or video) and flags:

- **Missing/incorrect PPE**: helmet, vest, goggles
- **Falls**: possible / high-confidence / sustained (with a 10-second alert)

It combines three YOLO models (fall detection, PPE, goggles) with a MediaPipe
pose-based sanity check (is the helmet actually *on the head*, not just
present in the frame?), then feeds everything into a rule engine that decides
violations and fall alerts per worker, tracked across frames.

## Project files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — upload image/video, live dashboard, alerts, report download |
| `pose_analyzer.py` | MediaPipe wrapper: `PersonPoseEstimator`, `PPEPositionVerifier` (is PPE worn correctly), `FallPoseAnalyzer` (does the pose look like a fall) |
| `rule_engine.py` | `RuleEngine` — turns YOLO + pose signals into Violations / Fall Status, tracks fall duration per worker across frames |
| `nlp_report.py` | Summarizes the worker log, generates an Arabic/English safety report via Gemini, exports it to PDF |

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Get the model weights

You need three trained YOLO `.pt` files:

- `best_fall.pt` — fall detection (class 0 = Fall-Detected, class 1 = Person)
- `ppe.pt` — helmet/vest (`{0: helmet, 1: no helmet, 2: no vest, 3: person, 4: vest}`)
- `best_Googles.pt` — goggles (class 0 = goggles, class 1 = no goggles)

Open `app.py` and point these three constants at wherever your `.pt` files
actually live:

```python
FALL_MODEL_PATH = "D:\\HEMAYA\\best_fall.pt"
PPE_MODEL_PATH = "D:\\HEMAYA\\ppe.pt"
GOGGLES_MODEL_PATH = "D:\\HEMAYA\\best_Googles.pt"
```

## 3. (Optional) Set up the Gemini report

Only needed if you plan to use the **"Generate Safety Report"** button. Pick
one of:

- Environment variable:
  ```bash
  # Windows PowerShell
  $env:GEMINI_API_KEY="your_key_here"
  # Windows CMD
  set GEMINI_API_KEY=your_key_here
  ```
- Or a `.streamlit/secrets.toml` file next to `app.py`:
  ```toml
  GEMINI_API_KEY = "your_key_here"
  ```

Get a key at <https://aistudio.google.com/apikey>.

PDF export needs Arabic shaping to render Arabic correctly — already covered
by `arabic-reshaper` + `python-bidi` in `requirements.txt`. Without them,
English PDFs still work, but Arabic text will render as disconnected/
reversed letters.

## 4. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` by default.

## How to use it

1. **Upload** an image or a video (`jpg/jpeg/png/mp4/avi/mov`).
2. **Image** → click *Run Detection* → see the annotated frame.
3. **Video** → adjust the two sliders if you want:
   - *Process every N frames* (higher = faster, choppier updates)
   - *Resize width for processing* (lower = faster)
   → click *▶ Run Video* to start the live feed + dashboard.
4. While it runs you'll see, per worker: helmet/vest/goggles status, current
   violations, and fall status. A sustained fall (≥10s) triggers a red
   ALERT box, an entry in the alerts log, and an audible beep.
5. After a run, expand **📊 Worker Summary** for a per-worker rollup, then
   pick a language and hit **🪄 Generate Report** to get an AI-written safety
   report, downloadable as PDF (or Markdown if the PDF Arabic dependencies
   aren't installed).
6. **⬇️ Download Full Worker Log (CSV)** exports every processed frame's raw
   readings for later analysis.

## Notes

- Fall alert threshold is 10 seconds by default — change it via
  `RuleEngine(fall_alert_seconds=...)` in `load_pose_pipeline()` in `app.py`.
- Worker IDs shown in the UI ("Worker 4") are small sequential numbers, not
  the raw ByteTrack IDs — they're remapped so the count on screen always
  matches what's actually visible.
- `st.session_state.worker_log` / `alerts_log` persist for the whole
  Streamlit session — refreshing the page clears them.
