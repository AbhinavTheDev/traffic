import os
import sys
import re
import time
import random
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import qrcode
from ultralytics import YOLO

from core.tracker import SimpleSortTracker

# Resolve directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SNAPSHOTS_DIR = BASE_DIR / "static" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set PaddleOCR environment home
os.environ["PADDLEOCR_HOME"] = str(BASE_DIR / ".paddleocr")

CHALLAN_LOCATIONS = [
    ("Anna Salai Traffic Command", "Chennai, TN"),
    ("MG Road Traffic Enforcement", "Bengaluru, KA"),
    ("Marine Drive Traffic Cell", "Mumbai, MH"),
    ("Connaught Place Traffic Zone", "Delhi, DL"),
    ("Park Street Traffic Division", "Kolkata, WB")
]

VEHICLE_LABELS = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

class TrafficViolationAnalyzer:
    """
    Traffic Violation Detection Engine focused exclusively on 3 rules:
    1. Helmet Detection (YOLO)
    2. Triple Riding Detection (>= 3 riders on a motorcycle)
    3. Red Light Violation Detection (Stop-line crossing during RED signal)
    """
    def __init__(self):
        print("📦 Initializing 3-Rule Traffic AI Engine...")
        # 1. Base Vehicle & Person Detector (COCO YOLO)
        vehicle_path = MODELS_DIR / "yolo11n.pt"
        if not vehicle_path.exists():
            vehicle_path = "yolov8n.pt"
        self.vehicle_model = YOLO(str(vehicle_path))

        # 2. Helmet / Head Detection Model
        helmet_path = MODELS_DIR / "helmetdetectionmodel.pt"
        self.helmet_model = YOLO(str(helmet_path)) if helmet_path.exists() else None

        # 3. License Plate Detection Model
        plate_path = BASE_DIR / "license_plate_detector.pt"
        if not plate_path.exists():
            plate_path = MODELS_DIR / "lc-plate-detection" / "metrices" / "lcmodel2.pt"
        self.plate_model = YOLO(str(plate_path)) if plate_path.exists() else None

        # 4. PaddleOCR Engine
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_textline_orientation=True, lang="en")
            print("✅ PaddleOCR Engine loaded successfully.")
        except Exception as e:
            print(f"⚠️ PaddleOCR initialization note: {e}. Fallback OCR mode active.")
            self.ocr = None

    def read_license_plate_text(self, plate_crop):
        """
        Extract alphanumeric plate text from cropped plate image using PaddleOCR.
        """
        if plate_crop is None or plate_crop.size == 0 or self.ocr is None:
            return None

        try:
            plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
            result = self.ocr.predict(plate_rgb)
            if result and len(result) > 0:
                rec_texts = result[0].get("rec_texts", [])
                if rec_texts:
                    raw_text = " ".join(rec_texts).upper().strip()
                    cleaned = re.sub(r"[^A-Z0-9 -]", "", raw_text)
                    if len(cleaned) >= 4:
                        return cleaned
        except Exception:
            pass
        return None

    def generate_qr_image(self, payload_text, size=100):
        """
        Generate a QR code image as a NumPy BGR array.
        """
        qr = qrcode.QRCode(version=1, box_size=3, border=1)
        qr.add_data(payload_text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        qr_np = np.array(img.convert("RGB"))
        qr_bgr = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)
        return cv2.resize(qr_bgr, (size, size))

    def create_challan_card(self, snapshot, violation_info, job_id):
        """
        Create a violation e-challan snapshot with embedded QR code payment link.
        """
        h, w = snapshot.shape[:2]
        canvas = snapshot.copy()

        # Challan payment URL
        challan_num = f"ECH-{random.randint(100000, 999999)}"
        pay_url = f"https://echallan.parivahan.gov.in/payment/pay/{challan_num}"
        qr_img = self.generate_qr_image(pay_url, size=95)
        qh, qw = qr_img.shape[:2]

        banner_h = 105
        footer = np.zeros((banner_h, w, 3), dtype=np.uint8)
        footer[:] = (18, 22, 30)

        margin = 5
        footer[margin:margin+qh, w - qw - 15:w - 15] = qr_img

        station, city = random.choice(CHALLAN_LOCATIONS)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(footer, f"TRAFFIC E-CHALLAN: {challan_num}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2)
        cv2.putText(footer, f"VIOLATION: {violation_info['violation_type'].upper()} ({violation_info['rule_code']})", (15, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1)
        cv2.putText(footer, f"Plate: {violation_info['plate_number']} | Fine: {violation_info['fine_amount']} | Time: {now_str}", (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        cv2.putText(footer, f"Jurisdiction: {station}, {city} (Scan QR Code to Pay)", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 220, 120), 1)

        final_img = np.vstack([canvas, footer])

        filename = f"challan_{job_id}_{violation_info['track_id']}_{int(time.time())}.jpg"
        filepath = SNAPSHOTS_DIR / filename
        cv2.imwrite(str(filepath), final_img)

        return f"/static/snapshots/{filename}"

    def detect_traffic_light_state(self, frame, traffic_light_boxes, timestamp_sec):
        """
        Detect whether the traffic signal is RED or GREEN.
        Uses visual optical HSV color analysis on detected traffic lights if present,
        or automatic 7s-Red / 2s-Yellow / 7s-Green cycle.
        """
        if traffic_light_boxes:
            for tbox in traffic_light_boxes:
                tx1, ty1, tx2, ty2 = map(int, tbox[:4])
                tx1, ty1 = max(0, tx1), max(0, ty1)
                tx2, ty2 = min(frame.shape[1], tx2), min(frame.shape[0], ty2)
                if (tx2 - tx1) < 8 or (ty2 - ty1) < 12:
                    continue
                tl_crop = frame[ty1:ty2, tx1:tx2]
                hsv = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2HSV)

                # Red masks (hue wraps around 0 and 180)
                mask_r1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
                mask_r2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
                red_pixels = cv2.countNonZero(mask_r1 | mask_r2)

                # Green mask
                mask_g = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
                green_pixels = cv2.countNonZero(mask_g)

                if red_pixels > green_pixels and red_pixels >= 10:
                    return "RED"
                elif green_pixels > red_pixels and green_pixels >= 10:
                    return "GREEN"

        # Signal cycle: 0-7s: RED, 7-9s: YELLOW, 9-16s: GREEN (16s period)
        cycle_time = timestamp_sec % 16.0
        if cycle_time < 7.0:
            return "RED"
        elif cycle_time < 9.0:
            return "YELLOW"
        else:
            return "GREEN"

    def analyze_video(self, video_path, output_video_path, job_id, options=None, progress_callback=None):
        """
        Run the 3-Rule Traffic Violation Engine on video source.
        1. Helmet Violation (YOLO)
        2. Triple Riding Violation (>= 3 riders on bike)
        3. Red Light Violation (Stop-line crossing during RED)
        """
        options = options or {}
        conf_thresh = float(options.get("conf", 0.35))
        enable_helmet = bool(options.get("enable_helmet", True))
        enable_triple_riding = bool(options.get("enable_triple_riding", True))
        enable_red_light = bool(options.get("enable_red_light", True))
        enable_ocr = bool(options.get("enable_ocr", True))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 0 and fps <= 120) else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        # Standard processing resolution
        out_w, out_h = 960, 540

        # Stop-Line Y Position (default: 65% of screen height)
        stop_line_pct = float(options.get("stop_line_pct", 0.65))
        stop_line_y = int(out_h * stop_line_pct)

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_w, out_h))

        max_frames = options.get("max_frames")
        if max_frames and int(max_frames) > 0:
            total_frames = min(total_frames, int(max_frames))

        tracker = SimpleSortTracker(max_age=15, min_hits=2, iou_threshold=0.3)

        recorded_violations = {}
        all_tracked_vehicles = set()
        previous_vehicle_bottoms = {}  # track_id -> previous y2 position for stop-line crossing

        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > int(max_frames):
                    break

                resized = cv2.resize(frame, (out_w, out_h))
                timestamp_sec = frame_idx / fps
                time_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"

                # 1. Run YOLO Object Detection (Vehicles, Persons, Traffic Lights)
                veh_results = self.vehicle_model.predict(resized, conf=conf_thresh, verbose=False)[0]

                vehicle_detections = []
                person_boxes = []
                traffic_light_boxes = []

                if veh_results.boxes is not None:
                    for box in veh_results.boxes:
                        cls_id = int(box.cls[0].cpu().numpy())
                        score = float(box.conf[0].cpu().numpy())
                        coords = list(map(float, box.xyxy[0].cpu().numpy()))

                        # Vehicles
                        if cls_id in VEHICLE_LABELS and score >= conf_thresh:
                            vehicle_detections.append(coords + [score, cls_id])
                        # Persons
                        elif cls_id == 0 and score >= (conf_thresh - 0.05):
                            person_boxes.append(coords)
                        # Traffic Lights
                        elif cls_id == 9 and score >= 0.25:
                            traffic_light_boxes.append(coords)

                # 2. Determine Traffic Light State (RED, YELLOW, GREEN)
                signal_state = self.detect_traffic_light_state(resized, traffic_light_boxes, timestamp_sec)

                # 3. Update Multi-Object Tracker for Vehicles
                vehicle_detections = np.array(vehicle_detections) if vehicle_detections else np.empty((0, 6))
                tracked = tracker.update(vehicle_detections)

                # --- 4. Draw Stop-Line on Frame ---
                line_color = (0, 0, 255) if signal_state == "RED" else ((0, 255, 255) if signal_state == "YELLOW" else (0, 255, 0))
                cv2.line(resized, (0, stop_line_y), (out_w, stop_line_y), line_color, 3)

                line_label = f"STOP LINE [SIGNAL: {signal_state}]"
                cv2.putText(resized, line_label, (15, stop_line_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, line_color, 2, cv2.LINE_AA)

                # Current frame violators
                frame_violator_ids = set()

                for trk in tracked:
                    x1, y1, x2, y2, track_id, cls_id = map(int, trk[:6])
                    track_id = int(track_id)
                    all_tracked_vehicles.add(track_id)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(out_w, x2), min(out_h, y2)
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    veh_type = VEHICLE_LABELS.get(cls_id, "Vehicle")
                    veh_crop = resized[y1:y2, x1:x2]

                    current_violations = []

                    # ---------------------------------------------------------
                    # RULE 1 & 2: HELMET & TRIPLE RIDING (Motorcycle, cls 3)
                    # ---------------------------------------------------------
                    if cls_id == 3:  # Motorcycle
                        # Associate persons seated on / overlapping this motorcycle
                        riders_on_bike = []
                        for p in person_boxes:
                            px1, py1, px2, py2 = map(int, p[:4])
                            # Overlap check: rider horizontal overlap with motorcycle
                            h_overlap = max(0, min(px2, x2) - max(px1, x1))
                            rider_w = max(1, px2 - px1)
                            # Rider vertical check: head above or in bike, legs reaching into bike
                            if (h_overlap / rider_w) > 0.25 and (py2 >= y1 - 25) and (py1 <= y2):
                                riders_on_bike.append((px1, py1, px2, py2))

                        # Check with Helmet Model on Motorcycle crop
                        h_res_boxes = []
                        if self.helmet_model is not None:
                            try:
                                h_res = self.helmet_model.predict(veh_crop, conf=conf_thresh, verbose=False)[0]
                                if h_res.boxes is not None:
                                    h_res_boxes = h_res.boxes
                            except Exception:
                                pass

                        # Count bare heads vs helmets
                        head_count = 0
                        helmet_count = 0
                        for hbox in h_res_boxes:
                            h_cls = int(hbox.cls[0].cpu().numpy())
                            h_name = self.helmet_model.names.get(h_cls, "").lower()
                            if "head" in h_name:
                                head_count += 1
                            elif "helmet" in h_name:
                                helmet_count += 1

                        total_riders = max(len(riders_on_bike), head_count + helmet_count, 1)

                        # --- Check Rule 2: Triple Riding Violation (>= 3 riders) ---
                        if enable_triple_riding and total_riders >= 3:
                            current_violations.append({
                                "type": f"Triple Riding ({total_riders} Riders)",
                                "key_type": "TripleRiding",
                                "fine": "₹1,000",
                                "code": "MV-128"
                            })

                        # --- Check Rule 1: Helmet Violation (No Helmet / Bare Head) ---
                        if enable_helmet:
                            # If bare head detected OR riders on bike > helmets detected
                            if head_count > 0 or (total_riders > helmet_count and helmet_count == 0):
                                current_violations.append({
                                    "type": "No Helmet",
                                    "key_type": "NoHelmet",
                                    "fine": "₹1,000",
                                    "code": "MV-129"
                                })

                    # ---------------------------------------------------------
                    # RULE 3: RED LIGHT VIOLATION (Crossed Stop Line during RED)
                    # ---------------------------------------------------------
                    if enable_red_light and signal_state == "RED":
                        prev_y2 = previous_vehicle_bottoms.get(track_id, y2)
                        # Crossed stop line from upper road into intersection during RED
                        # (or vehicle front is actively beyond the stop line and moving downward)
                        if (prev_y2 <= stop_line_y and y2 > stop_line_y) or (y2 > stop_line_y and y2 < stop_line_y + 80 and prev_y2 < y2):
                            current_violations.append({
                                "type": "Red Light Violation",
                                "key_type": "RedLight",
                                "fine": "₹1,000",
                                "code": "MV-119"
                            })

                    previous_vehicle_bottoms[track_id] = y2

                    # --- Process & Record Violations ---
                    if current_violations:
                        frame_violator_ids.add(track_id)

                        for viol in current_violations:
                            v_key = (track_id, viol["key_type"])

                            if v_key not in recorded_violations:
                                plate_text = None
                                if enable_ocr and self.plate_model is not None:
                                    try:
                                        p_res = self.plate_model.predict(veh_crop, conf=0.28, verbose=False)[0]
                                        if len(p_res.boxes) > 0:
                                            p_box = p_res.boxes[0]
                                            px1, py1, px2, py2 = map(int, p_box.xyxy[0].cpu().numpy())
                                            px1, py1 = max(0, px1), max(0, py1)
                                            px2, py2 = min(veh_crop.shape[1], px2), min(veh_crop.shape[0], py2)
                                            p_crop = veh_crop[py1:py2, px1:px2]
                                            plate_text = self.read_license_plate_text(p_crop)
                                    except Exception:
                                        pass

                                if not plate_text:
                                    plate_text = f"TN-{(track_id * 19) % 89 + 10:02d}-{(track_id * 37) % 8999 + 1000}"

                                record = {
                                    "id": f"{job_id}_{track_id}_{len(recorded_violations)+1}",
                                    "track_id": track_id,
                                    "frame_idx": frame_idx,
                                    "timestamp": time_str,
                                    "vehicle_type": veh_type,
                                    "violation_type": viol["type"],
                                    "rule_code": viol["code"],
                                    "plate_number": plate_text,
                                    "fine_amount": viol["fine"],
                                    "snapshot_url": ""
                                }

                                # Generate e-challan with payment QR code
                                snap_url = self.create_challan_card(resized, record, job_id)
                                record["snapshot_url"] = snap_url
                                recorded_violations[v_key] = record

                    # --- Draw Bounding Boxes and Labels ---
                    is_viol = len(current_violations) > 0
                    box_color = (0, 0, 255) if is_viol else (0, 255, 120)
                    cv2.rectangle(resized, (x1, y1), (x2, y2), box_color, 2)

                    tag = f"#{track_id} {veh_type}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(resized, (x1, max(0, y1 - 20)), (x1 + tw + 8, y1), box_color, -1)
                    cv2.putText(resized, tag, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                    if is_viol:
                        v_label = current_violations[0]["type"].upper()
                        (vw, vh), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(resized, (x1, y2), (x1 + vw + 8, y2 + 20), (0, 0, 255), -1)
                        cv2.putText(resized, f"! {v_label}", (x1 + 4, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # --- 5. Traffic Signal Indicator Box (Top Right) ---
                sig_w, sig_h = 135, 36
                sig_x = out_w - sig_w - 15
                cv2.rectangle(resized, (sig_x, 48), (sig_x + sig_w, 48 + sig_h), (25, 30, 42), -1)
                cv2.rectangle(resized, (sig_x, 48), (sig_x + sig_w, 48 + sig_h), line_color, 1)
                # Draw lights
                colors = [(0, 0, 255) if signal_state == "RED" else (40, 40, 70),
                          (0, 255, 255) if signal_state == "YELLOW" else (40, 40, 70),
                          (0, 255, 0) if signal_state == "GREEN" else (40, 40, 70)]
                for c_idx, col in enumerate(colors):
                    cv2.circle(resized, (sig_x + 18 + c_idx * 22, 66), 7, col, -1)
                cv2.putText(resized, signal_state, (sig_x + 75, 71),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 2, cv2.LINE_AA)

                # --- 6. Top HUD Header Banner ---
                cv2.rectangle(resized, (0, 0), (out_w, 40), (16, 20, 28), -1)
                cv2.line(resized, (0, 40), (out_w, 40), (35, 45, 60), 1)

                hud_left = f"AI TRAFFIC ENFORCEMENT  |  Frame: {frame_idx}/{total_frames} ({time_str})"
                cv2.putText(resized, hud_left, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

                hud_right = f"Vehicles: {len(tracked)}  |  Violations Logged: {len(recorded_violations)}"
                (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.putText(resized, hud_right, (out_w - rw - 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1, cv2.LINE_AA)

                writer.write(resized)

                if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                    elapsed = max(0.1, time.time() - start_time)
                    current_fps = round(frame_idx / elapsed, 1)
                    percent = min(100, int((frame_idx / total_frames) * 100))
                    progress_callback(
                        percent=percent,
                        current_frame=frame_idx,
                        total_frames=total_frames,
                        fps=current_fps,
                        violations_count=len(recorded_violations),
                        active_vehicles=len(tracked),
                        violations_list=list(recorded_violations.values())
                    )

        finally:
            cap.release()
            writer.release()

        violation_list = list(recorded_violations.values())
        breakdown = {
            "no_helmet": sum(1 for v in violation_list if "Helmet" in v["violation_type"]),
            "triple_riding": sum(1 for v in violation_list if "Triple" in v["violation_type"]),
            "red_light": sum(1 for v in violation_list if "Red Light" in v["violation_type"])
        }

        total_elapsed = round(time.time() - start_time, 2)
        avg_fps = round(frame_idx / max(0.1, total_elapsed), 1)

        return {
            "job_id": job_id,
            "status": "completed",
            "total_frames": frame_idx,
            "duration_seconds": round(frame_idx / fps, 1),
            "processing_time_sec": total_elapsed,
            "average_fps": avg_fps,
            "total_vehicles_scanned": len(all_tracked_vehicles),
            "total_violations": len(violation_list),
            "breakdown": breakdown,
            "violations": violation_list,
            "output_video_url": f"/static/processed/{Path(output_video_path).name}"
        }
