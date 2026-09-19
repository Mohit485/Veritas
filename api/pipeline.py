
from __future__ import annotations
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np

from compositor import Selection, build_target_mask, expand_mask, feather_mask, apply_blur, apply_pixelate, TrailBuffer, draw_motion_trail


class VideoPipeline:
    def __init__(self, detector_tracker, sam_refiner, effect: str = "pixelate",
                 sam_every_n_frames: int = 5, show_trail: bool = False, trail_length: int = 20):
        self.detector_tracker = detector_tracker
        self.sam_refiner = sam_refiner
        self.effect = effect
        self.sam_every_n_frames = max(1, sam_every_n_frames)
        self.trail_buffer = TrailBuffer(max_length=trail_length) if show_trail else None

    def run(self, video_path: str, output_path: str,
            selection: Selection = Selection(),
            max_seconds: Optional[float] = None):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video_path - is it a valid video file? {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(fps * max_seconds) if max_seconds else 0
        target_frames = min(total_frames, max_frames) if (total_frames > 0 and max_frames > 0) \
            else (max_frames or total_frames)

        self.detector_tracker.reset()
        if self.trail_buffer:
            self.trail_buffer.reset()
        last_masks: dict[int, np.ndarray] = {}

        # stderr goes to a temp FILE, not a pipe - a pipe deadlocks once
        # ffmpeg's cumulative progress output exceeds the OS pipe buffer
        # and nobody is draining it concurrently with the stdin writes
        # below. Confirmed this hangs forever with stderr=PIPE on a real
        # video; a temp file has no such buffer limit.
        stderr_file = tempfile.TemporaryFile()
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "pipe:0",
             "-i", video_path,
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
             "-shortest", "-movflags", "+faststart", output_path],
            stdin=subprocess.PIPE, stderr=stderr_file,
        )

        frame_idx = 0
        try:
            while True:
                if target_frames and frame_idx >= target_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                boxes = self.detector_tracker.process_frame(frame)

                # SAM refine cadence: on scheduled frames, refine everyone;
                # otherwise only refine ids with no mask yet (new arrivals).
                on_schedule = (frame_idx % self.sam_every_n_frames) == 0
                to_refine = boxes if on_schedule else {
                    tid: b for tid, b in boxes.items() if tid not in last_masks
                }
                if to_refine:
                    last_masks.update(self.sam_refiner.refine(frame, to_refine))
                last_masks = {tid: m for tid, m in last_masks.items() if tid in boxes}

                target = build_target_mask(last_masks, selection)
                if target is None:
                    expanded_target = expand_mask(target, expand_px=20) 
                    # 2. Feather edges
                    soft = feather_mask(expanded_target)
                    out_frame = frame
                else:
                    soft = feather_mask(target)
                    out_frame = apply_blur(frame, soft) if self.effect == "blur" else apply_pixelate(frame, soft)

                if self.trail_buffer:
                    trails = self.trail_buffer.update(boxes)
                    out_frame = draw_motion_trail(out_frame, trails)

                ffmpeg.stdin.write(np.ascontiguousarray(out_frame, dtype=np.uint8).tobytes())
                frame_idx += 1
                yield frame_idx, (target_frames or frame_idx)
        finally:
            cap.release()
            ffmpeg.stdin.close()
            ffmpeg.wait()
            stderr_file.seek(0)
            stderr_bytes = stderr_file.read()
            stderr_file.close()

        if ffmpeg.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ffmpeg.returncode}): {stderr_bytes.decode(errors='replace')}")

        print(f"Done - {frame_idx} frames processed. Saved to {output_path}")