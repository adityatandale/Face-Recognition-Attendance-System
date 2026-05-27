"""
Face Recognition Attendance System - Backend
=============================================
Requirements:
    pip install flask flask-cors face_recognition opencv-python numpy pandas openpyxl

Run:
    python backend.py

Folder structure:
    face_attendance/
    ├── backend.py
    ├── frontend.html
    ├── known_faces/       ← put <Name>.jpg images here to register people
    │   ├── Alice.jpg
    │   └── Bob.jpg
    ├── exports/           ← attendance CSV/XLSX files saved here
    └── attendance.db      ← SQLite database (auto-created)
"""

import os
import sqlite3
import csv
import json
import base64
import io
import logging
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd
try:
    import face_recognition  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'face_recognition'. Install it with `pip install face_recognition`."
    ) from exc
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR     = os.path.join(BASE_DIR, "known_faces")
EXPORTS_DIR   = os.path.join(BASE_DIR, "exports")
DB_PATH       = os.path.join(BASE_DIR, "attendance.db")
TOLERANCE     = 0.5          # lower = stricter match
FRAME_SCALE   = 0.5          # downscale frame for faster processing
LOG_LEVEL     = logging.INFO

os.makedirs(KNOWN_DIR,   exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow requests from the HTML frontend

# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL UNIQUE,
                added_on  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES students(id),
                name       TEXT    NOT NULL,
                date       TEXT    NOT NULL,
                time       TEXT    NOT NULL,
                status     TEXT    NOT NULL DEFAULT 'Present',
                UNIQUE(student_id, date)      -- one record per person per day
            );
        """)
    log.info("Database ready at %s", DB_PATH)


# ─── Face Encoding Cache ───────────────────────────────────────────────────────

known_encodings: list[np.ndarray] = []
known_names:     list[str]        = []


def load_known_faces():
    """
    Scan known_faces/ folder. Each file should be named <Full Name>.jpg/.png.
    Populates the in-memory encoding cache.
    """
    global known_encodings, known_names
    known_encodings, known_names = [], []

    supported = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(supported)]

    if not files:
        log.warning("No face images found in %s", KNOWN_DIR)
        return

    for fname in files:
        name = os.path.splitext(fname)[0]
        path = os.path.join(KNOWN_DIR, fname)
        img  = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)

        if not encs:
            log.warning("No face detected in %s — skipping", fname)
            continue

        known_encodings.append(encs[0])
        known_names.append(name)

        # Ensure student exists in DB
        with get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO students (name, added_on) VALUES (?, ?)",
                (name, datetime.now().isoformat(timespec="seconds"))
            )

    log.info("Loaded %d known face(s): %s", len(known_names), known_names)


# ─── Core Recognition Logic ────────────────────────────────────────────────────

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Convert a base64-encoded image string to an OpenCV BGR array."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image_to_base64(img_bgr: np.ndarray) -> str:
    """Convert an OpenCV BGR image to a base64 PNG string."""
    _, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def process_frame(img_bgr: np.ndarray) -> dict:
    """
    Detect and identify faces in a single frame.
    Returns annotated image + list of recognised names.
    """
    small = cv2.resize(img_bgr, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations  = face_recognition.face_locations(rgb, model="hog")
    encodings  = face_recognition.face_encodings(rgb, locations)

    detected   = []
    annotated  = img_bgr.copy()
    scale_inv  = 1.0 / FRAME_SCALE

    for enc, loc in zip(encodings, locations):
        matches    = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
        distances  = face_recognition.face_distance(known_encodings, enc)

        name       = "Unknown"
        confidence = 0.0

        if matches and len(distances):
            best_idx = int(np.argmin(distances))
            if matches[best_idx]:
                name       = known_names[best_idx]
                confidence = round((1 - distances[best_idx]) * 100, 1)

        detected.append({"name": name, "confidence": confidence})

        # Scale bounding box back to original resolution
        top, right, bottom, left = [int(v * scale_inv) for v in loc]
        color = (0, 200, 80) if name != "Unknown" else (0, 60, 220)

        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{name}  {confidence}%" if name != "Unknown" else "Unknown"
        cv2.rectangle(annotated, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
        cv2.putText(annotated, label, (left + 5, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    return {"annotated": annotated, "detected": detected}


def mark_attendance(name: str) -> dict:
    """
    Insert an attendance record for today if not already present.
    Returns {"marked": True/False, "message": str}.
    """
    if name == "Unknown":
        return {"marked": False, "message": "Face not recognised"}

    today     = date.today().isoformat()
    now_time  = datetime.now().strftime("%H:%M:%S")

    with get_db() as conn:
        student = conn.execute("SELECT id FROM students WHERE name=?", (name,)).fetchone()
        if not student:
            return {"marked": False, "message": f"'{name}' not registered in DB"}

        existing = conn.execute(
            "SELECT id FROM attendance WHERE student_id=? AND date=?",
            (student["id"], today)
        ).fetchone()

        if existing:
            return {"marked": False, "message": f"{name} already marked today"}

        conn.execute(
            "INSERT INTO attendance (student_id, name, date, time, status) VALUES (?,?,?,?,?)",
            (student["id"], name, today, now_time, "Present")
        )

    log.info("Marked attendance: %s at %s", name, now_time)
    return {"marked": True, "message": f"Attendance marked for {name}"}


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return jsonify({
        "status":        "running",
        "known_people":  len(known_names),
        "names":         known_names,
        "server_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.post("/api/reload_faces")
def api_reload():
    load_known_faces()
    return jsonify({"message": "Reloaded", "count": len(known_names), "names": known_names})


@app.post("/api/register")
def api_register():
    """
    Register a new person from a base64 image.
    Body: { "name": "Alice", "image": "data:image/jpeg;base64,..." }
    """
    data  = request.get_json(force=True)
    name  = data.get("name", "").strip()
    b64   = data.get("image", "")

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not b64:
        return jsonify({"error": "Image is required"}), 400

    img  = decode_base64_image(b64)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)

    if not encs:
        return jsonify({"error": "No face detected in the image"}), 422

    # Save image
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    path = os.path.join(KNOWN_DIR, f"{safe_name}.jpg")
    cv2.imwrite(path, img)

    load_known_faces()  # Refresh cache
    return jsonify({"message": f"'{safe_name}' registered successfully", "total": len(known_names)})


@app.post("/api/recognize")
def api_recognize():
    """
    Accepts a base64 frame, detects faces, marks attendance.
    Body: { "image": "data:image/jpeg;base64,..." }
    Returns: { "annotated": "data:image/png;...", "results": [...] }
    """
    data = request.get_json(force=True)
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"error": "No image provided"}), 400

    img    = decode_base64_image(b64)
    result = process_frame(img)

    outcomes = []
    for det in result["detected"]:
        att = mark_attendance(det["name"])
        outcomes.append({**det, **att})

    return jsonify({
        "annotated": encode_image_to_base64(result["annotated"]),
        "results":   outcomes,
        "count":     len(outcomes),
    })


@app.get("/api/attendance")
def api_attendance():
    """
    Query params: ?date=YYYY-MM-DD  (default: today)
                  ?all=1            (return all records)
    """
    all_records = request.args.get("all", "0") == "1"
    filter_date = request.args.get("date", date.today().isoformat())

    with get_db() as conn:
        if all_records:
            rows = conn.execute(
                "SELECT name, date, time, status FROM attendance ORDER BY date DESC, time DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name, date, time, status FROM attendance WHERE date=? ORDER BY time",
                (filter_date,)
            ).fetchall()

    return jsonify([dict(r) for r in rows])


@app.get("/api/students")
def api_students():
    with get_db() as conn:
        rows = conn.execute("SELECT name, added_on FROM students ORDER BY name").fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/stats")
def api_stats():
    today = date.today().isoformat()
    with get_db() as conn:
        total_students  = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        today_present   = conn.execute(
            "SELECT COUNT(*) FROM attendance WHERE date=?", (today,)
        ).fetchone()[0]
        total_records   = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        recent          = conn.execute(
            "SELECT name, time FROM attendance WHERE date=? ORDER BY time DESC LIMIT 5", (today,)
        ).fetchall()

    return jsonify({
        "total_students": total_students,
        "today_present":  today_present,
        "today_absent":   total_students - today_present,
        "total_records":  total_records,
        "recent_entries": [dict(r) for r in recent],
        "date":           today,
    })


@app.get("/api/export")
def api_export():
    """
    Export attendance to XLSX.
    Query params: ?date=YYYY-MM-DD  or  ?all=1
    """
    all_records = request.args.get("all", "0") == "1"
    filter_date = request.args.get("date", date.today().isoformat())

    with get_db() as conn:
        if all_records:
            rows = conn.execute(
                "SELECT name, date, time, status FROM attendance ORDER BY date DESC, time DESC"
            ).fetchall()
            filename = "attendance_all.xlsx"
        else:
            rows = conn.execute(
                "SELECT name, date, time, status FROM attendance WHERE date=? ORDER BY time",
                (filter_date,)
            ).fetchall()
            filename = f"attendance_{filter_date}.xlsx"

    df   = pd.DataFrame([dict(r) for r in rows], columns=["name", "date", "time", "status"])
    path = os.path.join(EXPORTS_DIR, filename)
    df.to_excel(path, index=False, engine="openpyxl")

    return send_file(path, as_attachment=True, download_name=filename)


@app.delete("/api/attendance/<int:record_id>")
def api_delete_record(record_id):
    with get_db() as conn:
        conn.execute("DELETE FROM attendance WHERE id=?", (record_id,))
    return jsonify({"message": "Record deleted"})


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    load_known_faces()
    log.info("Starting server on http://localhost:5000")
    log.info("Open frontend.html in your browser to use the dashboard")
    app.run(host="0.0.0.0", port=5000, debug=False)