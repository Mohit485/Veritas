from __future__ import annotations
import base64
import glob
import json
import os
import shutil
import time
import uuid
from enum import Enum
from typing import Optional

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from compositor import Selection, draw_labeled_boxes
from pipeline import VideoPipeline
from perception import DetectorTracker, SamRefiner

TMP_DIR = "/tmp/backlot"
os.makedirs(TMP_DIR, exist_ok=True)
STALE_UPLOAD_SECONDS = 3600

class Effect(str, Enum):
    blur = "blur"
    pixelate = "pixelate"

class Scope(str, Enum):
    selected = "selected"
    background = "background"

app = FastAPI(title="Redaction Pipeline API")

# Single global instance creation
_detector = None
_sam = None
_load_error = None

try:
    _detector = DetectorTracker(
        model_path=os.environ.get("YOLO_WEIGHTS", "yolo11n.onnx")
    )
    _sam = SamRefiner(
        encoder_path=os.environ.get("SAM_ENCODER_WEIGHTS", "sam_encoder.onnx"),
        decoder_path=os.environ.get("SAM_DECODER_WEIGHTS", "sam_decoder.onnx")
    )
except Exception as e:
    _load_error = str(e)


def _parse_classes(classes: str) -> list[int]:
    ids = [int(x) for x in classes.split(",") if x.strip().lstrip("-").isdigit()]
    return ids or [0]


def _cleanup_stale_uploads():
    now = time.time()
    for path in glob.glob(os.path.join(TMP_DIR, "*_in.*")):
        try:
            if now - os.path.getmtime(path) > STALE_UPLOAD_SECONDS:
                os.remove(path)
        except OSError:
            pass


def _resolve_video(video: Optional[UploadFile], job_id: Optional[str]) -> tuple[str, str]:
    if video is not None:
        _cleanup_stale_uploads()
        new_job_id = str(uuid.uuid4())[:8]
        video_path = os.path.join(TMP_DIR, f"{new_job_id}_in{os.path.splitext(video.filename)[1]}")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        return new_job_id, video_path

    if job_id:
        matches = glob.glob(os.path.join(TMP_DIR, f"{job_id}_in.*"))
        if matches:
            return job_id, matches[0]
        raise HTTPException(404, f"No stored video for job_id '{job_id}' - it may have expired or been deleted")

    raise HTTPException(400, "Provide either a video file, or a job_id from a previous call")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": _detector is not None and _sam is not None,
        "error": _load_error,
    }


@app.delete("/video/{job_id}")
def delete_video(job_id: str):
    matches = glob.glob(os.path.join(TMP_DIR, f"{job_id}_*"))
    if not matches:
        raise HTTPException(404, f"No stored files for job_id '{job_id}'")
    for path in matches:
        os.remove(path)
    return {"status": "deleted", "job_id": job_id}


@app.post("/preview")
async def preview(
    video: Optional[UploadFile] = File(None),
    job_id: Optional[str] = Form(None),
    second: float = Form(0.0),
    classes: str = Form("0"),
):
    if _detector is None:
        raise HTTPException(503, f"Pipeline not ready: {_load_error}")
    _detector.classes = set(_parse_classes(classes))
    job_id, video_path = _resolve_video(video, job_id)
            
    def event_stream():
        try:
            _detector.reset()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open the video")

            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            frame_number = max(0, round(second * fps))

            frame, boxes = None, {}
            try:
                for i in range(frame_number + 1):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Video is shorter than {second}s (fps={fps:.1f})")
                    boxes = _detector.process_frame(frame)
                    yield json.dumps({"type": "progress", "current": i + 1, "total": frame_number + 1}) + "\n"
            finally:
                cap.release()

            annotated = draw_labeled_boxes(frame, boxes)
            ok, encoded = cv2.imencode(".jpg", annotated)
            if not ok:
                raise RuntimeError("Failed to encode preview image")
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            yield json.dumps({"type": "result", "image_b64": image_b64}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers={"X-Job-Id": job_id})


@app.post("/process")
async def process_video(
    video: Optional[UploadFile] = File(None),
    job_id: Optional[str] = Form(None),
    effect: Effect = Form(Effect.pixelate),
    scope: Scope = Form(Scope.selected),
    selected_ids: str = Form(""),
    classes: str = Form("0"),
    max_seconds: Optional[float] = Form(None),
    sam_every_n_frames: int = Form(5),
    show_trail: bool = Form(False),
):
    if scope == Scope.selected and not selected_ids.strip():
        raise HTTPException(400, "scope='selected' requires selected_ids - call /preview first to find them")
    if _detector is None or _sam is None:
        raise HTTPException(503, f"Pipeline not ready: {_load_error}")
    
    _detector.reset()
    _detector.classes = set(_parse_classes(classes))

    job_id, video_path = _resolve_video(video, job_id)
    output_path = os.path.join(TMP_DIR, f"{job_id}_out.mp4")

    def event_stream():
        try:
            start_time = time.perf_counter()
            ids = {int(x) for x in selected_ids.split(",") if x.strip().lstrip("-").isdigit()}
            selection = Selection(scope=scope.value, mode="include", ids=ids)

            pipeline = VideoPipeline(_detector, _sam, effect=effect.value,
                                      sam_every_n_frames=sam_every_n_frames, show_trail=show_trail)
            
            total_frames_processed = 0
            for n, total in pipeline.run(video_path, output_path, selection=selection, max_seconds=max_seconds):
                total_frames_processed = n
                yield json.dumps({"type": "progress", "current": n, "total": total}) + "\n"

            elapsed_seconds = round(time.perf_counter() - start_time, 2)
            fps = round(total_frames_processed / max(0.001, elapsed_seconds), 2)
            sam_refine_calls = (total_frames_processed + sam_every_n_frames - 1) // sam_every_n_frames if sam_every_n_frames else 0

            with open(output_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("ascii")

            metrics = {
                "total_frames": total_frames_processed,
                "elapsed_seconds": elapsed_seconds,
                "fps": fps,
                "sam_refine_calls": sam_refine_calls,
                "active_ids": len(ids)
            }

            yield json.dumps({
                "type": "result", 
                "video_b64": video_b64, 
                "metrics": metrics
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
        finally:
            pass

    return StreamingResponse(event_stream(), media_type="application/x-ndjson", headers={"X-Job-Id": job_id})