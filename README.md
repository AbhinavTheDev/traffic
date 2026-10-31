# 🚦 Smart Traffic Violation Detection & Management System

An AI-powered computer vision and traffic management system designed to detect traffic violations, track vehicles, read license plates, and simulate dynamic traffic signals using deep learning models.

The system is fully modular, supports both **CPU and GPU** environments, and includes pre-trained weights and test video clips so you can run and test it immediately.

---

## 🧠 Core Features

- 🎯 **Helmet Violation Detection** (`scr/helmet_detect.py`)
  - Detects riders with and without helmets using YOLOv8.
  - Highlights violators with bounding boxes and live violation counts on screen.
  - Supports full-frame scanning or Region of Interest (ROI) scanning.

- 🚗 **Vehicle & Seatbelt Violation Detection** (`scr/the real+lc+seatbelt+ocr+qr.py`)
  - Identifies cars and motorcycles using YOLO.
  - Detects whether car drivers/passengers are wearing seatbelts.
  - Automatically flags violations in real time.

- 🔍 **License Plate Recognition & OCR** (`scr/the real+lc+seatbelt+ocr+qr.py`)
  - Locates vehicle license plates using a fine-tuned YOLO detector.
  - Extracts alphanumeric plate numbers using **PaddleOCR**.

- 📱 **QR Code E-Challan Generator**
  - Generates instant QR code payment links embedded onto violation snapshots.
  - Optional automated alerts via Telegram Bot.

- 🚦 **Dynamic 4-Lane Signal Handling & Speed Estimation** (`scr/signal_handling.py`)
  - Simulates 4-lane traffic signals (Red, Yellow, Green cycles).
  - Tracks vehicles across lanes using an IoU tracking algorithm.
  - Calculates real-time vehicle speed (km/h) based on pixel displacement.

---

## 💻 Hardware Requirements (CPU vs GPU)

- **GPU is NOT required!** The system runs smoothly on standard modern CPUs using PyTorch's CPU runtime.
- **CPU Performance**: Typical inference takes ~150–350ms per frame on CPU (~5–10 FPS), ideal for testing with pre-recorded videos.
- **CUDA / GPU (Optional)**: If an NVIDIA GPU with CUDA is present, PyTorch will automatically use it for 30–60+ FPS real-time webcam inference.

---

## 🗂️ Project Structure

```text
trafficfinal/
├── models/                               # Pre-trained model weights
│   ├── helmetdetectionmodel.pt           # Helmet & head detection YOLO model
│   ├── seatbeltmodel.pt                  # Seatbelt detection YOLO model
│   ├── yolo11n.pt                        # Base vehicle detection model (COCO)
│   └── lc-plate-detection/               # License plate model metrics & weights
├── license_plate_detector.pt             # Dedicated license plate detection weights
├── scr/                                  # Source scripts
│   ├── helmet_detect.py                  # Helmet violation detection
│   ├── signal_handling.py                # 4-lane signal & speed tracking
│   └── the real+lc+seatbelt+ocr+qr.py    # Combined vehicle, seatbelt, plate, OCR & QR
├── datafortesting-and-trainning/
│   └── testing/                          # Sample test traffic videos
│       ├── british_highway_traffic.mp4
│       ├── traffic_detection.mp4
│       ├── trafficCrowdVideo.mp4
│       ├── traffic_video.avi
│       └── vid1.mp4
├── triple_rider_test.mkv                 # Test video for helmet & motorcycle violations
├── red_light_test.mkv                    # Test video for red light / intersection testing
├── dynamic-signal-handling               # Root launcher for signal handling
├── requriments.txt                       # Project dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sairam-s0/Smart-Traffic-Violation-Detection-System.git
cd trafficfinal
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

**For CPU (Recommended if you don't have an NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requriments.txt
```

**For NVIDIA GPU (CUDA):**
```bash
pip install -r requriments.txt
```

---

## 🚀 Running & Testing with Videos

### 0. 🌐 Interactive Web Dashboard (Recommended)
Upload any traffic video or choose from sample clips, monitor detection in real-time, view side-by-side annotated video outputs, and inspect generated e-challans with QR payment codes.

```bash
python app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

- 📤 **Drag & Drop Video Upload**: Upload custom MP4, MKV, AVI, MOV videos.
- 🎯 **Pre-Loaded Sample Test Clips**: 1-click testing with included video clips.
- ⚡ **Live Processing Feedback**: Real-time progress bar, processing FPS, and live violation ticker.
- 📊 **Violation Analytics**: Scanned vehicle count, violation breakdown (Helmet, Seatbelt, Speeding).
- 🎬 **Annotated Video Player**: High-definition video with bounding boxes, tracking IDs, and speed HUD.
- 📱 **Interactive E-Challan Modal**: View official traffic violation tickets with scannable UPI/payment QR codes.
- 📥 **Export Reports**: Download the annotated MP4 video or export the violation log as CSV.

---

### 1. Helmet Detection
Detects motorcycle riders and flags those without helmets.

```powershell
# Run with default test video (triple_rider_test.mkv)
python scr/helmet_detect.py

# Replay video in an infinite loop
python scr/helmet_detect.py --source triple_rider_test.mkv --loop

# Test with your own downloaded video
python scr/helmet_detect.py --source path/to/your_video.mp4

# Run on live webcam (0 = default webcam)
python scr/helmet_detect.py --source 0

# Adjust confidence threshold (default: 0.35)
python scr/helmet_detect.py --source triple_rider_test.mkv --conf 0.5
```

**Controls**: Press `q` inside the video display window to stop playback.

---

### 2. Seatbelt, License Plate & OCR Violation Detection
Detects cars and bikes, inspects for seatbelts/helmets, locates license plates, extracts numbers with OCR, and embeds a QR payment link.

```powershell
# Run with default sample traffic video
python "scr/the real+lc+seatbelt+ocr+qr.py"

# Test with a specific traffic video
python "scr/the real+lc+seatbelt+ocr+qr.py" --source datafortesting-and-trainning/testing/traffic_detection.mp4

# Run with your own downloaded video
python "scr/the real+lc+seatbelt+ocr+qr.py" --source red_light_test.mkv

# Run on live webcam
python "scr/the real+lc+seatbelt+ocr+qr.py" --source 0
```

*(Note: Telegram alerts are disabled by default unless you provide your bot token inside the script).*

---

### 3. Dynamic Traffic Signal & Speed Tracking
Simulates a multi-lane traffic intersection with dynamic red/yellow/green light timing and vehicle speed estimation.

```powershell
# Runs 4-lane traffic signal simulation using local testing videos
python scr/signal_handling.py
```

---

## 📦 Defaults Provided Out-of-the-Box

| Feature | Default Video Source | Default Model Weights |
| :--- | :--- | :--- |
| **Helmet Detection** | [`triple_rider_test.mkv`](./triple_rider_test.mkv) | [`models/helmetdetectionmodel.pt`](./models/helmetdetectionmodel.pt) |
| **Vehicle Detection** | [`datafortesting-and-trainning/testing/traffic_detection.mp4`](./datafortesting-and-trainning/testing/traffic_detection.mp4) | [`models/yolo11n.pt`](./models/yolo11n.pt) |
| **Seatbelt Detection** | Tested per vehicle crop | [`models/seatbeltmodel.pt`](./models/seatbeltmodel.pt) |
| **License Plate Detection** | Tested per vehicle crop | [`license_plate_detector.pt`](./license_plate_detector.pt) |
| **4-Lane Traffic Simulation** | 4 videos in `datafortesting-and-trainning/testing/` | Built-in tracker + `yolo11n.pt` |

---

## 📊 Model Evaluation Summary

| **Model** | **Precision** | **Recall** | **mAP@0.5** | **mAP@0.5:0.95** | **Notes** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **License Plate** | ~0.925 | ~0.950 | ~0.967 | ~0.815 | High precision and reliable localization |
| **Seatbelt Detection** | ~0.840 | ~0.800 | ~0.850 | ~0.450 | Solid detection under clear windshield conditions |
| **Helmet Detection** | ~0.880 | ~0.870 | ~0.910 | ~0.620 | Distinguishes helmet vs bare head |

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
