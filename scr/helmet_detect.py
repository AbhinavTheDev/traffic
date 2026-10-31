import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Resolve base directories
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL = MODELS_DIR / "helmetdetectionmodel.pt"
DEFAULT_VIDEO = BASE_DIR / "triple_rider_test.mkv"

parser = argparse.ArgumentParser(description="Helmet Detection using YOLOv8")
parser.add_argument("--source", type=str,
                    default=str(DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else 0),
                    help="Path to video file or webcam index (default: triple_rider_test.mkv or 0)")
parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help="Path to helmet YOLO model (.pt)")
parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
parser.add_argument("--full-frame", action="store_true", default=True,
                    help="Detect across full frame instead of fixed ROI (default: True)")
parser.add_argument("--roi", action="store_true",
                    help="Enable ROI cropping (50:300) for top head area")
parser.add_argument("--loop", action="store_true",
                    help="Loop video continuously when it reaches the end")
args = parser.parse_args()

# Find the video file if relative path provided
video_source = args.source
if not str(video_source).isdigit():
    # If not absolute, search in current dir, BASE_DIR, or testing dir
    candidate = Path(video_source)
    if not candidate.exists():
        if (BASE_DIR / video_source).exists():
            video_source = str(BASE_DIR / video_source)
        elif (BASE_DIR / "datafortesting-and-trainning" / "testing" / video_source).exists():
            video_source = str(BASE_DIR / "datafortesting-and-trainning" / "testing" / video_source)
        else:
            print(f"❌ Error: Video file not found: {video_source}")
            print(f"   Checked in: {os.getcwd()}, {BASE_DIR}")
            exit(1)
    else:
        video_source = str(candidate.resolve())
else:
    video_source = int(video_source)

# Load the helmet detection model
print(f"📦 Loading Helmet Model: {args.model}")
helmet_model = YOLO(args.model)

# Open video source
print(f"🎥 Opening Video Source: {video_source}")
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Error: Unable to open video source: {video_source}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if (fps and fps > 0 and fps < 120) else 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not str(video_source).isdigit() else 0
delay = 1 if str(video_source).isdigit() else max(1, int(1000 / fps))

print(f"▶ Playing: {total_frames} frames @ {fps:.1f} FPS (press 'q' in the video window to stop)")

# Define Region of Interest (ROI) bounds if --roi is requested
use_roi = args.roi
roi_y1, roi_y2 = 50, 300

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        if args.loop and frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
        if frame_idx == 0:
            print("❌ Failed to read initial frame from video.")
        else:
            print(f"\n✅ Video playback completed! ({frame_idx} frames processed)")
        break

    frame_idx += 1

    # Resize for consistency
    resized = cv2.resize(frame, (640, 480))

    # Determine detection region
    if use_roi:
        roi = resized[roi_y1:roi_y2, :]
        offset_y = roi_y1
    else:
        roi = resized
        offset_y = 0

    # Run helmet detection
    results = helmet_model.predict(roi, conf=args.conf, verbose=False)

    boxes = []
    helmet_count = 0
    no_helmet_count = 0

    for result in results:
        if result.boxes is not None:
            for det in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2 = map(int, det[:4])
                conf = float(det[4])
                cls_id = int(det[5])
                label = helmet_model.model.names.get(cls_id, "Unknown").lower()

                # Map box coords back to full frame
                y1 += offset_y
                y2 += offset_y
                boxes.append([x1, y1, x2, y2, conf, label])

    # Apply Non-Maximum Suppression (NMS)
    nms_threshold = 0.4
    if boxes:
        nms_indices = cv2.dnn.NMSBoxes(
            bboxes=[box[:4] for box in boxes],
            scores=[box[4] for box in boxes],
            score_threshold=args.conf,
            nms_threshold=nms_threshold
        )

        if nms_indices is not None and len(nms_indices) > 0:
            if isinstance(nms_indices, (list, tuple)):
                nms_indices = np.array(nms_indices).flatten()

            for i in nms_indices:
                x1, y1, x2, y2, conf, label = boxes[i]
                is_helmet = (label == "helmet")
                if is_helmet:
                    helmet_count += 1
                else:
                    no_helmet_count += 1

                color = (0, 255, 0) if is_helmet else (0, 0, 255)
                tag = "Helmet" if is_helmet else "No Helmet"
                cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized, f"{tag} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show live info banner on top
    info_text = f"Frame: {frame_idx}/{total_frames} | Helmets: {helmet_count} | Violations: {no_helmet_count}"
    cv2.putText(resized, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Periodic console status
    if frame_idx % 30 == 0 or frame_idx == 1:
        print(f"\r⏳ Progress: [{frame_idx}/{total_frames}] | Active: {helmet_count} Helmets, {no_helmet_count} Violations", end="", flush=True)

    # Show the frame
    cv2.imshow("Helmet Detection", resized)

    # Press 'q' to exit
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print(f"\n⏹ Stopped by user at frame {frame_idx}.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
