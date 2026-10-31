import os
import io
import csv
import uuid
import time
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

from core.analyzer import TrafficViolationAnalyzer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"
SNAPSHOTS_DIR = STATIC_DIR / "snapshots"
TEMPLATES_DIR = BASE_DIR / "templates"
TESTING_DIR = BASE_DIR / "datafortesting-and-trainning" / "testing"

for folder in [UPLOADS_DIR, PROCESSED_DIR, SNAPSHOTS_DIR, TEMPLATES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Traffic Violation Detection System - 3 Rules")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if (BASE_DIR / "datafortesting-and-trainning").exists():
    app.mount("/datafortesting-and-trainning", StaticFiles(directory=str(BASE_DIR / "datafortesting-and-trainning")), name="datafortesting")

SAMPLE_VIDEOS = [
    {
        "id": "triple_rider",
        "name": "Triple Rider & Helmet Test Clip",
        "description": "Dense motorcycle test clip with multiple riders without helmets & 3+ riders on bike.",
        "path": str(BASE_DIR / "triple_rider_test.mkv"),
        "url": "/static/samples/triple_rider_test.mkv",
        "type": "Helmet & Triple Riding"
    },
    {
        "id": "red_light",
        "name": "Intersection Red Light Test Clip",
        "description": "Urban intersection clip with vehicles approaching and crossing the stop line.",
        "path": str(BASE_DIR / "red_light_test.mkv"),
        "url": "/static/samples/red_light_test.mkv",
        "type": "Red Light Violation"
    },
    {
        "id": "traffic_detection",
        "name": "Multi-Vehicle Traffic Highway",
        "description": "Multi-lane traffic with cars, trucks, motorcycles, and stop-line enforcement.",
        "path": str(TESTING_DIR / "traffic_detection.mp4"),
        "url": "/datafortesting-and-trainning/testing/traffic_detection.mp4",
        "type": "Mixed Multi-Vehicle"
    }
]

# Ensure symlinks/copies in static/samples
SAMPLES_DIR = STATIC_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
for sample in [BASE_DIR / "triple_rider_test.mkv", BASE_DIR / "red_light_test.mkv"]:
    if sample.exists():
        target = SAMPLES_DIR / sample.name
        if not target.exists():
            try:
                os.link(str(sample), str(target))
            except Exception:
                shutil.copy2(str(sample), str(target))

analyzer_instance: Optional[TrafficViolationAnalyzer] = None
analyzer_lock = threading.Lock()

def get_analyzer():
    global analyzer_instance
    with analyzer_lock:
        if analyzer_instance is None:
            analyzer_instance = TrafficViolationAnalyzer()
        return analyzer_instance

jobs = {}
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template = jinja_env.get_template("index.html")
    return template.render(sample_videos=SAMPLE_VIDEOS)

@app.get("/api/sample-videos")
async def list_sample_videos():
    available = [s for s in SAMPLE_VIDEOS if Path(s["path"]).exists()]
    return {"samples": available}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload MP4, MKV, AVI, MOV, or WEBM.")

    unique_name = f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = UPLOADS_DIR / unique_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "filename": file.filename,
        "file_path": str(save_path),
        "file_url": f"/static/uploads/{unique_name}"
    }

def run_analysis_worker(job_id: str, video_path: str, options: dict):
    output_filename = f"analyzed_{job_id}.mp4"
    output_video_path = PROCESSED_DIR / output_filename

    analyzer = get_analyzer()

    def progress_callback(percent, current_frame, total_frames, fps, violations_count, active_vehicles, violations_list):
        jobs[job_id]["percent"] = percent
        jobs[job_id]["current_frame"] = current_frame
        jobs[job_id]["total_frames"] = total_frames
        jobs[job_id]["fps"] = fps
        jobs[job_id]["violations_count"] = violations_count
        jobs[job_id]["active_vehicles"] = active_vehicles
        jobs[job_id]["violations_list"] = violations_list

    try:
        results = analyzer.analyze_video(
            video_path=video_path,
            output_video_path=output_video_path,
            job_id=job_id,
            options=options,
            progress_callback=progress_callback
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["percent"] = 100
        jobs[job_id]["results"] = results
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/api/analyze")
async def start_analysis(payload: dict):
    source = payload.get("video_source", "")

    video_path = None
    for sample in SAMPLE_VIDEOS:
        if sample["id"] == source:
            video_path = sample["path"]
            break

    if not video_path:
        video_path = source

    if not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "percent": 0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0.0,
        "violations_count": 0,
        "active_vehicles": 0,
        "violations_list": [],
        "results": None,
        "error": None,
        "video_path": video_path,
        "start_time": time.time()
    }

    thread = threading.Thread(
        target=run_analysis_worker,
        args=(job_id, video_path, payload),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "status": "started"}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "percent": job["percent"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "fps": job["fps"],
        "violations_count": job["violations_count"],
        "active_vehicles": job["active_vehicles"],
        "violations_list": job["violations_list"][-15:],
        "error": job.get("error")
    }

@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] == "failed":
        return JSONResponse(status_code=500, content={"status": "failed", "error": job["error"]})
    if job["status"] != "completed":
        return JSONResponse(status_code=202, content={"status": "processing", "percent": job["percent"]})

    return job["results"]

@app.get("/api/export/{job_id}/csv")
async def export_violations_csv(job_id: str):
    if job_id not in jobs or not jobs[job_id].get("results"):
        raise HTTPException(status_code=404, detail="Job results not found")

    violations = jobs[job_id]["results"].get("violations", [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Challan ID", "Vehicle Track ID", "Vehicle Type", "Timestamp",
        "Violation Rule", "Motor Vehicle Act Code", "License Plate",
        "Fine Amount", "Snapshot Image"
    ])

    for v in violations:
        writer.writerow([
            v.get("id"),
            v.get("track_id"),
            v.get("vehicle_type"),
            v.get("timestamp"),
            v.get("violation_type"),
            v.get("rule_code", "MV-ACT"),
            v.get("plate_number"),
            v.get("fine_amount"),
            v.get("snapshot_url")
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=traffic_violations_{job_id}.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    get_analyzer()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
