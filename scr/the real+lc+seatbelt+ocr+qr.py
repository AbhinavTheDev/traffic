import os
from pathlib import Path
import argparse
import random
from datetime import datetime
import cv2
import torch
import qrcode
import numpy as np
import requests
from paddleocr import PaddleOCR
from ultralytics import YOLO

os.environ['PADDLEOCR_HOME'] = './.paddleocr'

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Default Model Paths
DEFAULT_VEHICLE_MODEL = MODELS_DIR / "yolo11n.pt" if (MODELS_DIR / "yolo11n.pt").exists() else "yolov8n.pt"
DEFAULT_HELMET_MODEL = MODELS_DIR / "helmetdetectionmodel.pt"
DEFAULT_PLATE_MODEL = BASE_DIR / "license_plate_detector.pt" if (BASE_DIR / "license_plate_detector.pt").exists() else MODELS_DIR / "lc-plate-detection" / "metrices" / "lcmodel2.pt"
DEFAULT_SEATBELT_MODEL = MODELS_DIR / "seatbeltmodel.pt"

# Default Sample Video
DEFAULT_SAMPLE_VIDEO = BASE_DIR / "datafortesting-and-trainning" / "testing" / "traffic_detection.mp4"

parser = argparse.ArgumentParser(description="Traffic Violation Detection (Seatbelt, Helmet, License Plate, OCR, QR)")
parser.add_argument("--source", type=str,
                    default=str(DEFAULT_SAMPLE_VIDEO if DEFAULT_SAMPLE_VIDEO.exists() else 0),
                    help="Path to video file or webcam index (default: testing/traffic_detection.mp4 or 0)")
parser.add_argument("--vehicle-model", type=str, default=str(DEFAULT_VEHICLE_MODEL), help="Path to vehicle YOLO model")
parser.add_argument("--helmet-model", type=str, default=str(DEFAULT_HELMET_MODEL), help="Path to helmet YOLO model")
parser.add_argument("--plate-model", type=str, default=str(DEFAULT_PLATE_MODEL), help="Path to plate YOLO model")
parser.add_argument("--seatbelt-model", type=str, default=str(DEFAULT_SEATBELT_MODEL), help="Path to seatbelt YOLO model")
parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
args = parser.parse_args()

# Load models
print(f"📦 Loading Vehicle Model: {args.vehicle_model}")
vehicle_model = YOLO(args.vehicle_model)

print(f"📦 Loading Helmet Model: {args.helmet_model}")
helmet_model = YOLO(args.helmet_model)

print(f"📦 Loading License Plate Model: {args.plate_model}")
plate_model = YOLO(args.plate_model)

print(f"📦 Loading Seatbelt Model: {args.seatbelt_model}")
seatbelt_model = YOLO(args.seatbelt_model)

print("📦 Initializing PaddleOCR...")
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Constants
VEHICLE_CLASSES = [2, 3]  # 2: car, 3: motorcycle in COCO
HELMET_CONF = args.conf
PLATE_CONF = args.conf
SEATBELT_CONF = args.conf

TELEGRAM_BOT_TOKEN = "your bot token "
TELEGRAM_CHAT_ID = "your id s u can give in a list f have many"

# Locations for demo challan
places = [
    ("Chennai", "Teynampet Traffic Police Station, Anna Salai, Chennai – 600018"),
    ("Madurai", "Tallakulam Traffic Police Station, Madurai – 625002"),
    ("Coimbatore", "RS Puram Traffic Police Station, Coimbatore – 641002"),
    ("Trichy", "Trichy South Traffic Police Station, Near Main Bazaar, Trichy – 620001"),
    ("Salem", "Hasthampatti Traffic Police Station, Salem – 636007"),
]

# QR code generator
def generate_qr(text):
    qr = qrcode.QRCode(version=1, box_size=2, border=1)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_image = np.array(img.convert('RGB'))
    return qr_image

# Telegram alert function
def send_violation_alert(image, plate_text="Unknown"):
    place, station = random.choice(places)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fine_payment_link = f"https://traffic.tn.gov.in/payfine/{random.randint(10000,99999)}"

    qr_img = generate_qr(fine_payment_link)
    qr_img = cv2.resize(qr_img, (100, 100))

    h, w, _ = image.shape
    x_offset = w - qr_img.shape[1] - 10
    y_offset = h - qr_img.shape[0] - 10
    image[y_offset:y_offset+qr_img.shape[0], x_offset:x_offset+qr_img.shape[1]] = qr_img
    caption = f"""🚨 *Traffic Violation Detected* 🚨
*Violation:* No Helmet or No Seatbelt
*Time:* {now}
*Place:* {place}, Tamil Nadu
*Fine Amount:* ₹1000

*License Plate:* {plate_text}

 _To appeal: Call 1800-100-1234 or visit [www.traffic.tn.gov.in/appeal](https://traffic.tn.gov.in/appeal)_  

*Pay Fine At:*
{station}
"""

    if not TELEGRAM_BOT_TOKEN or "your" in TELEGRAM_BOT_TOKEN.lower():
        # Telegram token not configured, skip sending network request
        return

    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('violation.jpg', img_encoded.tobytes())}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        response = requests.post(url, files=files, data=data, timeout=5)

        if not response.ok:
            print("Failed to send Telegram alert:", response.text)
    except Exception as e:
        print("Telegram alert error:", e)

# Open video source (webcam index or video file)
video_source = args.source
if not str(video_source).isdigit():
    candidate = Path(video_source)
    if not candidate.exists():
        if (BASE_DIR / video_source).exists():
            video_source = str(BASE_DIR / video_source)
        elif (BASE_DIR / "datafortesting-and-trainning" / "testing" / video_source).exists():
            video_source = str(BASE_DIR / "datafortesting-and-trainning" / "testing" / video_source)
        else:
            print(f"❌ Error: Video file not found: {video_source}")
            exit(1)
    else:
        video_source = str(candidate.resolve())
else:
    video_source = int(video_source)

print(f"🎥 Opening Video Source: {video_source}")
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Error: Unable to access video source: {video_source}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if (fps and fps > 0 and fps < 120) else 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not str(video_source).isdigit() else 0
delay = 1 if str(video_source).isdigit() else max(1, int(1000 / fps))

print(f"▶ Playing: {total_frames} frames @ {fps:.1f} FPS (press 'q' in the video window to stop)")

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if frame_idx == 0:
            print("❌ Failed to read initial frame from video.")
        else:
            print(f"\n✅ Video playback completed! ({frame_idx} frames processed)")
        break

    frame_idx += 1

    frame = cv2.resize(frame, (640, 480))
    small = frame.copy()

    vehicle_results = vehicle_model.predict(small, conf=args.conf, verbose=False)[0]

    for box in vehicle_results.boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
        if cls_id in VEHICLE_CLASSES:
            # Detect car or motorcycle
            vehicle_crop = small[by1:by2, bx1:bx2]
            if vehicle_crop.size == 0:
                continue

            if cls_id == 2:  # Car detected
                # Seatbelt detection for cars
                seatbelt_results = seatbelt_model.predict(vehicle_crop, conf=SEATBELT_CONF, verbose=False)[0]
                seatbelt_detected = len(seatbelt_results.boxes) > 0

                if not seatbelt_detected:
                    cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(small, "No Seatbelt", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    plate_text = "Unknown"
                    # Number plate detection 
                    plate_results = plate_model.predict(vehicle_crop, conf=PLATE_CONF, verbose=False)[0]
                    for pbox in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                        gx1, gy1 = bx1 + px1, by1 + py1
                        gx2, gy2 = bx1 + px2, by1 + py2
                        cv2.rectangle(small, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            try:
                                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                plate_result = ocr.predict(plate_crop_rgb)
                                if plate_result and len(plate_result[0].get("rec_texts", [])) > 0:
                                    plate_text = " ".join(plate_result[0]["rec_texts"]).strip()
                                    cv2.putText(small, plate_text, (gx1, gy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print("OCR Error:", e)
                    # alerts
                    send_violation_alert(small.copy(), plate_text)

            elif cls_id == 3:  # Motorcycle detected
                # Helmet detection for motorcycles 
                helmet_results = helmet_model.predict(vehicle_crop, conf=HELMET_CONF, verbose=False)[0]
                helmet_detected = False
                for hbox in helmet_results.boxes:
                    h_cls = int(hbox.cls[0].cpu().numpy())
                    h_label = helmet_model.names.get(h_cls, "").lower()
                    if h_label == "helmet":
                        helmet_detected = True
                        break

                if not helmet_detected:
                    cv2.rectangle(small, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(small, "No Helmet", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    plate_text = "Unknown"
                    # Number plate detection 
                    plate_results = plate_model.predict(vehicle_crop, conf=PLATE_CONF, verbose=False)[0]
                    for pbox in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0].cpu().numpy())
                        gx1, gy1 = bx1 + px1, by1 + py1
                        gx2, gy2 = bx1 + px2, by1 + py2
                        cv2.rectangle(small, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                        plate_crop = vehicle_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            try:
                                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                plate_result = ocr.predict(plate_crop_rgb)
                                if plate_result and len(plate_result[0].get("rec_texts", [])) > 0:
                                    plate_text = " ".join(plate_result[0]["rec_texts"]).strip()
                                    cv2.putText(small, plate_text, (gx1, gy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print("OCR Error:", e)
                    # Send alert
                    send_violation_alert(small.copy(), plate_text)

    cv2.imshow('Traffic Monitoring', small)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
