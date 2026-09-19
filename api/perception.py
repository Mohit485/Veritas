from __future__ import annotations
import os
import cv2
import numpy as np
import onnxruntime as ort

# Dynamically find the absolute path of the directory where perception.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "yolo11n.onnx")
DEFAULT_SAM_ENCODER = os.path.join(BASE_DIR, "sam_encoder.onnx")
DEFAULT_SAM_DECODER = os.path.join(BASE_DIR, "sam_decoder.onnx")

COCO_PERSON = 0

class DetectorTracker:
    """YOLO ONNX Detector paired with NMS filtering and IoU spatial tracking."""
    def __init__(self, model_path: str = DEFAULT_YOLO_PATH, confidence: float = 0.3,
                 classes: list[int] | None = None, iou_threshold: float = 0.3):
        # Allow default ONNX thread handling inside Docker
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.confidence = confidence
        self.classes = set(classes if classes is not None else [COCO_PERSON])
        self.input_name = self.session.get_inputs()[0].name
        self.iou_threshold = iou_threshold
        
        # Tracking state
        self.next_track_id = 0
        self.tracked_objects: dict[int, tuple[int, int, int, int]] = {}
        
        # Pre-warm session graph instantly upon init
        self.warmup()

    def warmup(self):
        """Runs a dummy pass to compile ONNX computational graph on server boot."""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.process_frame(dummy_frame)
        self.reset()

    def reset(self):
        """Resets tracking state between video processing runs."""
        self.next_track_id = 0
        self.tracked_objects = {}

    def _compute_iou(self, boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def process_frame(self, frame: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]

        # Fast Resize & Preprocess for YOLO
        img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {self.input_name: img})[0]
        outputs = np.squeeze(outputs).T

        # Vectorized confidence filtering
        scores = np.max(outputs[:, 4:], axis=1)
        class_ids = np.argmax(outputs[:, 4:], axis=1)
        mask = (scores >= self.confidence) & np.isin(class_ids, list(self.classes))

        filtered_outputs = outputs[mask]
        filtered_scores = scores[mask]

        if len(filtered_outputs) == 0:
            self.tracked_objects = {}
            return {}

        boxes_list, confidences = [], []
        x_scale, y_scale = w / 640.0, h / 640.0

        # --- OPTION 3: Bounding Box Padding Factor (15% Expansion) ---
        pad_factor = 0.15 

        for row, score in zip(filtered_outputs, filtered_scores):
            cx, cy, bw, bh = row[0:4]
            
            # Convert to actual frame coordinates
            orig_w = bw * x_scale
            orig_h = bh * y_scale
            orig_x1 = (cx - bw / 2) * x_scale
            orig_y1 = (cy - bh / 2) * y_scale

            # Apply 15% padding around width and height
            pad_w = orig_w * pad_factor
            pad_h = orig_h * pad_factor

            padded_x1 = int(orig_x1 - pad_w)
            padded_y1 = int(orig_y1 - pad_h)
            padded_bw = int(orig_w + (2 * pad_w))
            padded_bh = int(orig_h + (2 * pad_h))

            boxes_list.append([padded_x1, padded_y1, padded_bw, padded_bh])
            confidences.append(float(score))

        # NMS Deduplication
        indices = cv2.dnn.NMSBoxes(boxes_list, confidences, self.confidence, 0.45)
        new_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes_list[idx]
                # Hard clamp expanded boxes so they don't exceed frame dimensions
                new_boxes.append((max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)))

        # IoU Spatial Tracking
        updated_tracks = {}
        unmatched_new_boxes = list(new_boxes)

        for track_id, old_box in self.tracked_objects.items():
            best_iou, best_match_idx = 0.0, -1
            for idx, candidate_box in enumerate(unmatched_new_boxes):
                iou = self._compute_iou(old_box, candidate_box)
                if iou > best_iou:
                    best_iou, best_match_idx = iou, idx

            if best_iou >= self.iou_threshold and best_match_idx != -1:
                updated_tracks[track_id] = unmatched_new_boxes.pop(best_match_idx)

        for box in unmatched_new_boxes:
            updated_tracks[self.next_track_id] = box
            self.next_track_id += 1

        self.tracked_objects = updated_tracks
        return self.tracked_objects


class SamRefiner:
    """Pure ONNX Implementation of MobileSAM with box displacement caching."""
    def __init__(self, encoder_path: str = DEFAULT_SAM_ENCODER,
                 decoder_path: str = DEFAULT_SAM_DECODER,
                 motion_threshold_px: float = 30.0): # <--- Set to 30.0 to limit SAM calls on CPU
        # Standard ONNX initialization without custom thread overrides
        self.encoder_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self.decoder_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
        
        self.encoder_input = self.encoder_session.get_inputs()[0].name
        self.decoder_inputs = [i.name for i in self.decoder_session.get_inputs()]
        
        self.motion_threshold_px = motion_threshold_px
        self._last_embeddings = None
        self._last_boxes: dict[int, tuple[int, int, int, int]] = {}
        
        # Pre-warm sessions
        self.warmup()
        
        # Pre-warm sessions
        self.warmup()

    def warmup(self):
        """Pre-runs encoder and decoder ONNX sessions to avoid runtime initialization latency."""
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_box = {0: (100, 100, 200, 200)}
        self.refine(dummy_frame, dummy_box, force_reencode=True)
        self._last_embeddings = None
        self._last_boxes = {}

    def _has_significant_motion(self, new_boxes: dict[int, tuple[int, int, int, int]]) -> bool:
        if not self._last_boxes or set(new_boxes.keys()) != set(self._last_boxes.keys()):
            return True

        for track_id, (x1, y1, x2, y2) in new_boxes.items():
            lx1, ly1, lx2, ly2 = self._last_boxes[track_id]
            displacement = max(abs(x1 - lx1), abs(y1 - ly1), abs(x2 - lx2), abs(y2 - ly2))
            if displacement > self.motion_threshold_px:
                return True

        return False

    def _preprocess_image(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        rgb = (rgb - mean) / std
        
        tensor = rgb.transpose(2, 0, 1)[None, :, :, :]
        return tensor, (h, w)

    def refine(self, frame: np.ndarray, boxes: dict[int, tuple[int, int, int, int]],
               force_reencode: bool = False) -> dict[int, np.ndarray]:
        if not boxes:
            return {}

        orig_h, orig_w = frame.shape[:2]

        should_encode = (
            force_reencode 
            or self._last_embeddings is None 
            or self._has_significant_motion(boxes)
        )

        if should_encode:
            # Directly preprocess original frame to 1024x1024 without double-resizing
            img_tensor, _ = self._preprocess_image(frame)
            self._last_embeddings = self.encoder_session.run(None, {self.encoder_input: img_tensor})[0]
            self._last_boxes = boxes.copy()

        embeddings = self._last_embeddings
        sx, sy = 1024.0 / orig_w, 1024.0 / orig_h
        masks: dict[int, np.ndarray] = {}

        for track_id, (x1, y1, x2, y2) in boxes.items():
            scaled_box = np.array([[[x1 * sx, y1 * sy, x2 * sx, y2 * sy]]], dtype=np.float32)
            
            raw_mask = self.decoder_session.run(
                None, 
                {
                    self.decoder_inputs[0]: embeddings,
                    self.decoder_inputs[1]: scaled_box
                }
            )[0]

            mask_data = (raw_mask[0, 0] > 0.0).astype(np.uint8)
            resized_mask = cv2.resize(mask_data, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            masks[track_id] = resized_mask

        return masks