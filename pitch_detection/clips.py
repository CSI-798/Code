"""Utilities for generating pitch clips with pose overlay."""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 165, 255),
    (255, 128, 0),
    (128, 255, 0),
]


def _parse_pose_keypoints(
    keypoints,
    confidences: Optional[List[float]],
) -> Tuple[List[Tuple[float, float]], Optional[List[float]]]:
    if not keypoints:
        return [], confidences

    points: List[Tuple[float, float]] = []
    parsed_confidences: List[float] = []

    first_item = keypoints[0] if isinstance(keypoints, list) and keypoints else None

    if isinstance(first_item, (list, tuple)):
        for item in keypoints:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            x, y = float(item[0]), float(item[1])
            points.append((x, y))
            if len(item) >= 3:
                parsed_confidences.append(float(item[2]))
    elif isinstance(first_item, (int, float)):
        numeric_keypoints = [float(v) for v in keypoints]
        if len(numeric_keypoints) >= 51 and len(numeric_keypoints) % 3 == 0:
            for i in range(0, len(numeric_keypoints), 3):
                points.append((numeric_keypoints[i], numeric_keypoints[i + 1]))
                parsed_confidences.append(numeric_keypoints[i + 2])
        elif len(numeric_keypoints) >= 34 and len(numeric_keypoints) % 2 == 0:
            for i in range(0, len(numeric_keypoints), 2):
                points.append((numeric_keypoints[i], numeric_keypoints[i + 1]))
        else:
            return [], confidences

    if parsed_confidences:
        return points, parsed_confidences
    return points, confidences


def _joint_valid(
    points: List[Tuple[float, float]],
    idx: int,
    frame_w: int,
    frame_h: int,
    confidences: Optional[List[float]],
    confidence_threshold: float,
) -> bool:
    x, y = points[idx]
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    if x <= 0 or y <= 0 or x >= frame_w or y >= frame_h:
        return False
    if confidences is not None and idx < len(confidences):
        confidence = float(confidences[idx])
        if not math.isfinite(confidence):
            return False
        return confidence >= confidence_threshold
    return True


def _draw_pose(
    frame,
    keypoints: List[float],
    confidences: Optional[List[float]],
    color: Tuple[int, int, int],
    confidence_threshold: float,
):
    frame_h, frame_w = frame.shape[:2]
    points, confidences = _parse_pose_keypoints(keypoints, confidences)
    if len(points) < 5:
        return

    max_limb_length = math.hypot(frame_w, frame_h) * 0.35

    for left, right in SKELETON_EDGES:
        if left >= len(points) or right >= len(points):
            continue
        if _joint_valid(points, left, frame_w, frame_h, confidences, confidence_threshold) and _joint_valid(
            points, right, frame_w, frame_h, confidences, confidence_threshold
        ):
            lx, ly = points[left]
            rx, ry = points[right]
            if math.hypot(lx - rx, ly - ry) <= max_limb_length:
                cv2.line(
                    frame,
                    (int(round(lx)), int(round(ly))),
                    (int(round(rx)), int(round(ry))),
                    color,
                    2,
                )

    for idx, (x, y) in enumerate(points):
        if _joint_valid(points, idx, frame_w, frame_h, confidences, confidence_threshold):
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, -1)


def _draw_all_poses(frame, poses: List[Dict], confidence_threshold: float):
    for idx, pose in enumerate(poses):
        color = PALETTE[idx % len(PALETTE)]
        keypoints = pose.get("keypoints", [])
        confidences = pose.get("confidence")
        _draw_pose(frame, keypoints, confidences, color, confidence_threshold)

        bbox = pose.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def generate_overlay_clips(
    predictions_file: str,
    pose_file: str,
    video_file: str,
    output_dir: str,
    padding_seconds: float = 2.0,
    max_clips: Optional[int] = None,
    confidence_threshold: float = 0.25,
    overlay: bool = True,
) -> List[str]:
    predictions_path = Path(predictions_file)
    pose_path = Path(pose_file)
    video_path = Path(video_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(predictions_path, "r") as f:
        prediction_data = json.load(f)
    with open(pose_path, "r") as f:
        pose_data = json.load(f)

    segments = prediction_data.get("segments", [])
    if max_clips is not None:
        segments = segments[: max(0, int(max_clips))]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = prediction_data.get("parameters", {}).get("fps", 30)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    padding_frames = int(max(0, padding_seconds) * fps)
    pose_frames = pose_data.get("poses", [])

    output_files: List[str] = []

    for idx, seg in enumerate(segments, 1):
        pitch_start = int(seg.get("start_frame", 0))
        pitch_end = int(seg.get("end_frame", 0))
        clip_start = max(0, pitch_start - padding_frames)
        clip_end = min(total_frames - 1, pitch_end + padding_frames)

        clip_prefix = "pitch_overlay" if overlay else "pitch_clip"
        clip_name = f"{clip_prefix}_{idx:03d}.mp4"
        clip_path = output_path / clip_name

        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"Could not create output video: {clip_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for frame_idx in range(clip_start, clip_end + 1):
            ok, frame = cap.read()
            if not ok:
                break

            if overlay:
                if 0 <= frame_idx < len(pose_frames):
                    frame_poses = pose_frames[frame_idx].get("poses", [])
                    if frame_poses:
                        _draw_all_poses(frame, frame_poses, confidence_threshold)

                if pitch_start <= frame_idx <= pitch_end:
                    cv2.rectangle(frame, (8, 8), (width - 8, height - 8), (0, 255, 0), 3)
                    cv2.putText(frame, "PITCH", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            writer.write(frame)

        writer.release()
        output_files.append(str(clip_path))

    cap.release()
    return output_files
