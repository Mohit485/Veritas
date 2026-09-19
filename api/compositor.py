"""
compositor.py
-------------
Which pixels get redacted, and how. Two effects only - blur and
pixelate - both legitimate anonymization techniques. Background
replacement was cut when this project narrowed from a general-purpose
"virtual background" tool to a purpose-built redaction tool: swapping
backgrounds isn't part of what redaction needs, and dropping it also
removed an entire class of bugs (the background-video file-handle leak)
along with the code that caused them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Literal
import cv2
import numpy as np


@dataclass
class Selection:
    """
    scope="selected"    -> target = specific track ids (mode="include"),
                            or everyone EXCEPT those ids (mode="exclude")
                            - redact only the people you pick. Default,
                            since this is the core redaction workflow.
    scope="background"  -> target = everything that is NOT a tracked
                            person/object - protect the tracked
                            subject(s), redact everyone/everything else
                            (e.g. "keep the interviewee sharp, blur all
                            bystanders").
    """
    scope: Literal["background", "selected"] = "selected"
    mode: Literal["include", "exclude"] = "include"
    ids: set[int] = field(default_factory=set)


def combine_masks(masks: dict[int, np.ndarray]) -> np.ndarray | None:
    combined = None
    for mask in masks.values():
        mask = mask.astype(np.uint8)
        combined = mask if combined is None else cv2.bitwise_or(combined, mask)
    return combined


def build_target_mask(masks: dict[int, np.ndarray], selection: Selection) -> np.ndarray | None:
    if not masks:
        return None
    if selection.scope == "background":
        union = combine_masks(masks)
        return None if union is None else (1 - union).astype(np.uint8)
    chosen_ids = selection.ids if selection.mode == "include" else set(masks.keys()) - selection.ids
    chosen = {oid: m for oid, m in masks.items() if oid in chosen_ids}
    return combine_masks(chosen)

def expand_mask(mask: np.ndarray, expand_px: int = 15) -> np.ndarray:
    """Expands (dilates) the redaction mask by expand_px pixels."""
    if mask is None or expand_px <= 0:
        return mask
    # Create a circular kernel for smooth, rounded expansion
    kernel_size = expand_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def feather_mask(mask: np.ndarray, edge_softness: int = 7) -> np.ndarray:
    if edge_softness % 2 == 0:
        edge_softness += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (edge_softness, edge_softness), 0)


def _alpha3(mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(mask, 0, 1).astype(np.float32)
    return cv2.merge([alpha, alpha, alpha])


def apply_blur(frame: np.ndarray, target_mask: np.ndarray, blur_strength: int = 35) -> np.ndarray:
    if blur_strength % 2 == 0:
        blur_strength += 1
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    alpha = _alpha3(target_mask)
    return (blurred.astype(np.float32) * alpha + frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def apply_pixelate(frame: np.ndarray, target_mask: np.ndarray, block_size: int = 18) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = _alpha3(target_mask)
    return (mosaic.astype(np.float32) * alpha + frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)


class TrailBuffer:
    """
    Keeps the last K foot positions per tracked id across frames -
    optional, secondary feature. Visualizing movement paths of tracked
    subjects can help a reviewer understand context before deciding who
    to redact; not required for the core workflow.
    """
    def __init__(self, max_length: int = 20, max_frames_missing: int = 15):
        self.max_length = max_length
        self.max_frames_missing = max_frames_missing
        self._trails: dict[int, deque] = {}
        self._last_seen: dict[int, int] = {}
        self._frame_idx = 0

    def reset(self):
        self._trails.clear()
        self._last_seen.clear()
        self._frame_idx = 0

    def update(self, boxes: dict[int, tuple[int, int, int, int]]) -> dict[int, list[tuple[int, int]]]:
        self._frame_idx += 1
        for track_id, (x1, y1, x2, y2) in boxes.items():
            foot = (int((x1 + x2) / 2), int(y2))
            self._trails.setdefault(track_id, deque(maxlen=self.max_length)).append(foot)
            self._last_seen[track_id] = self._frame_idx
        expired = [tid for tid, seen in self._last_seen.items() if self._frame_idx - seen > self.max_frames_missing]
        for tid in expired:
            self._trails.pop(tid, None)
            self._last_seen.pop(tid, None)
        return {tid: list(pts) for tid, pts in self._trails.items()}


def draw_motion_trail(frame: np.ndarray, trails: dict[int, list[tuple[int, int]]],
                       color: tuple[int, int, int] = (60, 220, 255)) -> np.ndarray:
    out = frame.copy()
    for points in trails.values():
        n = len(points)
        if n < 2:
            continue
        overlay = out.copy()
        for i in range(1, n):
            thickness = max(1, int(2 + (i / n) * 4))
            cv2.line(overlay, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
        cv2.circle(out, points[-1], 5, color, -1, cv2.LINE_AA)
    return out


def draw_labeled_boxes(frame: np.ndarray, boxes: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    """
    Used by /preview - draws each tracked box with its id number
    labeled, so a caller can see which id corresponds to which person
    before choosing who to redact.

    Two things this deliberately gets right, because a naive version
    both breaks in practice:

    1. Label size scales with the FRAME's own resolution, not a fixed
       pixel size. A fixed size drawn on a 1080p source frame becomes
       illegibly small once a frontend downscales the image for
       display (e.g. to a few hundred px wide); the same fixed size on
       a small frame dwarfs a small, distant person's box.
    2. Label placement is edge-aware: it goes above the box by default,
       but flips to below the box when the box's top is close enough to
       the top of the frame that the label would be clipped off (a
       "big" box - i.e. one that reaches near the top edge - is exactly
       when this happens), and is clamped horizontally so it can never
       run off the left or right edge either (which is what happens to
       a small box sitting near the frame's edge).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    font_scale = max(0.45, min(1.4, min(h, w) / 480))
    thickness = max(1, round(font_scale * 2))
    box_thickness = max(2, round(font_scale * 2))

    for track_id, (x1, y1, x2, y2) in boxes.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), box_thickness)

        label = f"id {track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = max(4, round(font_scale * 6))

        # Try above the box, falling back to below if that would clip
        # off the top - then, regardless of which side was picked, clamp
        # hard to the frame bounds. The above/below choice alone isn't
        # sufficient: a box large enough to reach near BOTH the top and
        # bottom edges would still clip on whichever side gets picked,
        # so the final clamp is what actually guarantees correctness.
        label_h = th + pad * 2
        label_y1 = (y1 - label_h) if (y1 - label_h) >= 0 else y2
        label_y1 = max(0, min(label_y1, h - label_h))
        label_y2 = label_y1 + label_h
        text_y = label_y2 - pad

        label_w = tw + pad * 2
        label_x1 = max(0, min(x1, w - label_w))
        label_x2 = label_x1 + label_w

        cv2.rectangle(out, (label_x1, label_y1), (label_x2, label_y2), (40, 220, 40), -1)
        cv2.putText(out, label, (label_x1 + pad, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out