# 🎯 Face Recognition Attendance System

An automated attendance system that uses real-time face recognition to detect, identify, and mark attendance — no manual entry required. Built with Python, OpenCV, and a clean web dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📸 Features

- **Live face detection** via webcam with bounding boxes and confidence scores
- **Auto attendance marking** — scans every 2.5 seconds, marks once per person per day
- **Register new faces** from webcam capture or uploaded image
- **Dashboard** with today's present/absent counts and recent check-ins
- **Attendance records** filterable by date
- **Export to Excel (.xlsx)** — daily or full history
- **SQLite database** — zero setup, self-contained
- **REST API backend** — clean separation, easy to extend

---

## 🗂️ Project Structure

```
face_attendance/
├── backend.py          # Flask API server
├── frontend.html       # Web dashboard (open in browser)
├── known_faces/        # Drop face images here (Name.jpg)
├── exports/            # Exported Excel files saved here
└── attendance.db       # SQLite DB (auto-created on first run)
```

---

## ⚙️ Requirements

- Python 3.8+
- Webcam
- Windows / macOS / Linux

### Python Libraries

```
flask
flask-cors
opencv-python
numpy
pandas
openpyxl
face_recognition
dlib
```

---

## 🚀 Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/face-attendance-system.git
cd face-attendance-system
```

### Step 2 — Install Dependencies

```bash
pip install flask flask-cors opencv-python numpy pandas openpyxl
```

### Step 3 — Install dlib + face_recognition

`dlib` requires a C++ compiler and CMake before installing via pip.

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev
pip install dlib face_recognition
```

**macOS**
```bash
brew install cmake
pip install dlib face_recognition
```

**Windows**
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → select **"Desktop development with C++"**
2. Install [CMake](https://cmake.org/download/) → check **"Add to PATH"** during install
3. Restart terminal, then:
```bash
pip install dlib face_recognition
```

> **Windows pre-built wheel (if above fails):**
> Find your Python version with `python --version`, then grab the matching wheel from:
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x
> ```bash
> pip install <wheel-url>
> pip install face_recognition
> ```

---

## 📋 Usage

### 1. Add Known Faces

Place `.jpg` or `.png` images into the `known_faces/` folder.
Each filename becomes the person's name in the system.

```
known_faces/
├── Alice Johnson.jpg
├── Bob Smith.jpg
└── Raj Patel.png
```

> One face per image. Front-facing, good lighting works best.

### 2. Start the Backend

```bash
python backend.py
```

Expected output:
```
Database ready at .../attendance.db
Loaded 3 known face(s): ['Alice Johnson', 'Bob Smith', 'Raj Patel']
Starting server on http://localhost:5000
```

**Keep this terminal open.**

### 3. Open the Dashboard

Double-click `frontend.html` or open it in your browser.

If the webcam doesn't work from `file://`, serve it locally:
```bash
python -m http.server 8080
# Then open: http://localhost:8080/frontend.html
```

---

## 🖥️ Dashboard Pages

| Page | What it does |
|------|-------------|
| **Dashboard** | Today's stats, recent check-ins |
| **Recognize** | Live camera feed, auto-scan, recognition log |
| **Register Face** | Add a new person via camera or image upload |
| **Attendance** | View/filter records by date, export to Excel |
| **Students** | List of all registered people |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Server status and loaded faces |
| GET | `/api/stats` | Today's attendance counts |
| POST | `/api/recognize` | Detect faces in a base64 image |
| POST | `/api/register` | Register a new face |
| POST | `/api/reload_faces` | Reload known_faces/ folder |
| GET | `/api/attendance` | Get records (`?date=YYYY-MM-DD` or `?all=1`) |
| GET | `/api/students` | List all registered students |
| GET | `/api/export` | Download Excel file (`?date=` or `?all=1`) |
| DELETE | `/api/attendance/<id>` | Delete a specific record |

---

## 🛠️ Configuration

Edit these constants at the top of `backend.py`:

```python
TOLERANCE    = 0.5   # Face match strictness (lower = stricter, 0.4–0.6 recommended)
FRAME_SCALE  = 0.5   # Downscale factor for faster processing
```

Auto-scan interval (in `frontend.html`):
```javascript
const AUTO_INTERVAL_MS = 2500;  // milliseconds between scans
```

---

## 🧠 How It Works

1. Known face images are loaded and encoded into 128-dimension vectors on startup
2. Each video frame is downscaled and sent to the Flask backend
3. `face_recognition` detects face locations and computes encodings
4. Encodings are compared against known faces using Euclidean distance
5. Matches below the tolerance threshold are identified by name
6. A single attendance record is inserted per person per day (duplicate-safe)
7. The annotated frame is returned to the frontend for display

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Face Detection | `face_recognition` + `dlib` HOG model |
| Image Processing | `OpenCV` |
| Backend | `Flask` + `Flask-CORS` |
| Database | `SQLite` (via Python `sqlite3`) |
| Export | `pandas` + `openpyxl` |
| Frontend | Vanilla HTML/CSS/JS |

---

## 🐛 Common Issues

**`dlib` build fails**
→ Install CMake and C++ build tools first. See Installation → Step 3.

**Camera not working in browser**
→ Run `python -m http.server 8080` and open via `http://localhost:8080/frontend.html`

**"No face detected" on registration**
→ Ensure the image has one clear, front-facing face with good lighting.

**Server dot stays red (offline)**
→ Backend is not running. Make sure `python backend.py` is running in a separate terminal.

**Face not recognized despite being registered**
→ Lower the `TOLERANCE` value in `backend.py` (try `0.6`) or re-register with a clearer photo.

---

## 📄 License

This project is free to use.

---

## 🙌 Acknowledgements

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib](http://dlib.net/) by Davis King
- [OpenCV](https://opencv.org/)
