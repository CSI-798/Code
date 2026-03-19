"""
Video Preprocessing Module for Pitch Detection

This module handles the extraction of pose features and video features from input videos.
It includes:
1. YOLO pose estimation using YOLOv8-Pose
2. Motion feature extraction using frame differencing (image subtraction)
"""

import cv2
import torch
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List


class YOLOPoseExtractor:
    """Extract pose features from videos using YOLOv8-Pose"""
    
    def __init__(self, model_path: str = 'yolov8l-pose.pt', device: str = 'cuda', batch_size: int = 64):
        """
        Initialize YOLO pose extractor
        
        Args:
            model_path: Path to YOLOv8-Pose model weights
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Number of frames to process per YOLO inference call
        """
        self.device = device
        self.model_path = model_path
        self.batch_size = max(1, int(batch_size))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model weights not found: {model_path}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLOv8-Pose model from {model_path}")
            print(f"   Device: {device}")
            print(f"   Pose batch size: {self.batch_size}")
        except ImportError:
            raise ImportError("ultralytics package not found. Install with: pip install ultralytics")

    def _yolo_device(self):
        """Resolve ultralytics-compatible device spec."""
        if self.device.startswith('cuda') and torch.cuda.is_available():
            if ':' in self.device:
                try:
                    return int(self.device.split(':', 1)[1])
                except ValueError:
                    return 0
            return 0
        return 'cpu'

    def _select_primary_pose_index(
        self,
        frame_poses: List[Dict],
        previous_center: Optional[Tuple[float, float]]
    ) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        """Select a stable primary person index with confidence + temporal continuity."""
        if not frame_poses:
            return None, None

        best_idx = 0
        best_score = -float('inf')
        best_center = None

        for idx, pose in enumerate(frame_poses):
            bbox = pose.get('bbox')
            pose_score = float(pose.get('pose_score', 0.0))

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            else:
                cx = cy = 0.0
                area = 0.0

            continuity_bonus = 0.0
            if previous_center is not None:
                dist = ((cx - previous_center[0]) ** 2 + (cy - previous_center[1]) ** 2) ** 0.5
                continuity_bonus = -0.001 * dist

            score = pose_score + 0.00001 * area + continuity_bonus
            if score > best_score:
                best_score = score
                best_idx = idx
                best_center = (cx, cy)

        return best_idx, best_center

    def _extract_frame_poses(self, result) -> List[Dict]:
        """Extract all person poses from one YOLO result object."""
        frame_poses: List[Dict] = []
        if result is None:
            return frame_poses

        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return frame_poses

        keypoints_data = result.keypoints
        if not hasattr(keypoints_data, 'xy'):
            return frame_poses

        xy = keypoints_data.xy.cpu().numpy()  # Shape: (num_persons, 17, 2)
        conf_arr = (
            keypoints_data.conf.cpu().numpy()
            if hasattr(keypoints_data, 'conf') and keypoints_data.conf is not None
            else None
        )

        boxes_xyxy = None
        boxes_conf = None
        if hasattr(result, 'boxes') and result.boxes is not None:
            if hasattr(result.boxes, 'xyxy') and result.boxes.xyxy is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                boxes_conf = result.boxes.conf.cpu().numpy()

        for person_idx in range(xy.shape[0]):
            keypoints = xy[person_idx]  # Shape: (17, 2)
            keypoints_flat = keypoints.flatten().tolist()

            if conf_arr is not None and person_idx < len(conf_arr):
                confidence = conf_arr[person_idx].tolist()
            else:
                confidence = [1.0] * 17

            if boxes_xyxy is not None and person_idx < len(boxes_xyxy):
                bbox = boxes_xyxy[person_idx].tolist()
            else:
                valid_points = keypoints[np.any(keypoints > 0, axis=1)]
                if len(valid_points) > 0:
                    x1, y1 = valid_points.min(axis=0)
                    x2, y2 = valid_points.max(axis=0)
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                else:
                    bbox = None

            bbox_confidence = float(boxes_conf[person_idx]) if boxes_conf is not None and person_idx < len(boxes_conf) else None
            pose_score = float(np.mean(confidence)) if confidence else 0.0

            frame_poses.append({
                'keypoints': keypoints_flat,
                'confidence': confidence,
                'bbox': bbox,
                'bbox_confidence': bbox_confidence,
                'pose_score': pose_score
            })

        return frame_poses

    def _predict_pose_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: float,
    ) -> List:
        """Run batched YOLO inference with automatic CUDA OOM backoff."""
        if not frames:
            return []

        device = self._yolo_device()
        total = len(frames)
        cursor = 0
        target_batch = min(self.batch_size, total)
        all_results: List = []

        while cursor < total:
            current_batch = min(target_batch, total - cursor)

            while True:
                frame_slice = frames[cursor: cursor + current_batch]
                try:
                    results = self.model.predict(
                        source=frame_slice,
                        conf=conf_threshold,
                        classes=[0],
                        device=device,
                        batch=current_batch,
                        verbose=False,
                    )

                    if results is None:
                        normalized_results = [None] * len(frame_slice)
                    else:
                        normalized_results = list(results)
                        if len(normalized_results) < len(frame_slice):
                            normalized_results.extend([None] * (len(frame_slice) - len(normalized_results)))
                        elif len(normalized_results) > len(frame_slice):
                            normalized_results = normalized_results[:len(frame_slice)]

                    all_results.extend(normalized_results)
                    cursor += current_batch
                    break

                except RuntimeError as error:
                    message = str(error).lower()
                    is_oom = "out of memory" in message and "cuda" in message

                    if is_oom and device != 'cpu' and current_batch > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        target_batch = max(1, current_batch // 2)
                        print(f"⚠️  CUDA OOM at pose batch {current_batch}, retrying with batch {target_batch}")
                        current_batch = target_batch
                        continue

                    if is_oom and device != 'cpu' and current_batch == 1:
                        raise RuntimeError(
                            "CUDA out of memory even at pose batch size 1. "
                            "Try --device cpu or lower other GPU load."
                        ) from error

                    raise

        return all_results
    
    def extract_from_video(
        self, 
        video_path: str, 
        output_dir: str = 'pose_features',
        conf_threshold: float = 0.3,
        verbose: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Extract pose features from video file
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save pose features
            conf_threshold: Confidence threshold for pose detection
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (pose_data_list, metadata)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = video_path.stem
        
        if verbose:
            print(f"\n🎬 Extracting pose features from {video_name}")
            print(f"   Total frames: {total_frames:,}")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
        
        # Process video frames
        pose_data_list = []
        frame_number = 0
        frames_with_poses = 0
        total_people_detected = 0
        previous_primary_center = None
        frame_batch: List[np.ndarray] = []
        frame_index_batch: List[int] = []
        
        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames", disable=not verbose)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_batch.append(frame)
            frame_index_batch.append(frame_number)

            if len(frame_batch) >= self.batch_size:
                results = self._predict_pose_batch(
                    frames=frame_batch,
                    conf_threshold=conf_threshold,
                )

                for local_idx, frame_idx in enumerate(frame_index_batch):
                    result = results[local_idx] if results is not None and local_idx < len(results) else None
                    frame_poses = self._extract_frame_poses(result)

                    primary_pose_index, previous_primary_center = self._select_primary_pose_index(
                        frame_poses, previous_primary_center
                    )

                    if frame_poses:
                        frames_with_poses += 1
                        total_people_detected += len(frame_poses)

                    pose_data_list.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps if fps > 0 else 0,
                        'poses': frame_poses,
                        'primary_pose_index': primary_pose_index
                    })

                frame_batch = []
                frame_index_batch = []
            
            frame_number += 1
            pbar.update(1)

        if frame_batch:
            results = self._predict_pose_batch(
                frames=frame_batch,
                conf_threshold=conf_threshold,
            )

            for local_idx, frame_idx in enumerate(frame_index_batch):
                result = results[local_idx] if results is not None and local_idx < len(results) else None
                frame_poses = self._extract_frame_poses(result)

                primary_pose_index, previous_primary_center = self._select_primary_pose_index(
                    frame_poses, previous_primary_center
                )

                if frame_poses:
                    frames_with_poses += 1
                    total_people_detected += len(frame_poses)

                pose_data_list.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'poses': frame_poses,
                    'primary_pose_index': primary_pose_index
                })
        
        pbar.close()
        cap.release()
        
        if verbose:
            print(f"✅ Extracted poses from {len(pose_data_list):,} frames")
            print(f"   Frames with detected poses: {frames_with_poses:,} ({frames_with_poses/len(pose_data_list)*100:.1f}%)")
            if len(pose_data_list) > 0:
                print(f"   Avg people detected/frame: {total_people_detected / len(pose_data_list):.2f}")
        
        # Create output structure
        output_data = {
            'video_path': str(video_path.absolute()),
            'video_name': video_name,
            'metadata': {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps > 0 else 0
            },
            'poses': pose_data_list,
            'extraction_info': {
                'model': 'yolov8-pose',
                'conf_threshold': conf_threshold,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save to JSON
        output_path = output_dir / f"{video_name}_pose_features.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if verbose:
            print(f"💾 Saved pose features to: {output_path}")
        
        # Create summary
        summary = {
            'total_frames': len(pose_data_list),
            'frames_with_poses': frames_with_poses,
            'detection_rate': frames_with_poses / len(pose_data_list) if len(pose_data_list) > 0 else 0,
            'avg_people_per_frame': total_people_detected / len(pose_data_list) if len(pose_data_list) > 0 else 0
        }
        
        summary_path = output_dir / f"{video_name}_pose_features_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return pose_data_list, output_data['metadata']


class VideoFeatureExtractor:
    """Extract motion features using frame differencing (image subtraction)."""
    
    def __init__(self, device: str = 'cuda', feature_dim: int = 400, batch_size: int = 64):
        """
        Initialize video feature extractor
        
        Args:
            device: Device hint (kept for API compatibility)
            feature_dim: Output feature dimension for motion vectors
            batch_size: Unused (kept for CLI/API compatibility)
        """
        self.device = device
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.grid_h, self.grid_w = self._compute_grid_shape(feature_dim)
        
        print(f"🚀 Initializing motion feature extractor...")
        print(f"   Device: {device}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Motion grid: {self.grid_h} x {self.grid_w}")
        print(f"✅ Motion extractor ready")

    @staticmethod
    def _compute_grid_shape(feature_dim: int) -> Tuple[int, int]:
        """Compute a compact 2D grid whose flattened size is >= feature_dim."""
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        grid_h = int(np.floor(np.sqrt(feature_dim)))
        grid_h = max(1, grid_h)
        grid_w = int(np.ceil(feature_dim / grid_h))
        return grid_h, grid_w

    def _extract_motion_feature(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> np.ndarray:
        """Compute one motion feature vector from consecutive grayscale frames."""
        diff = cv2.absdiff(curr_gray, prev_gray)

        # Light denoising + stable scaling
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        resized = cv2.resize(diff, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)
        vector = resized.astype(np.float32).reshape(-1) / 255.0

        if vector.size < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:vector.size] = vector
            return padded
        return vector[: self.feature_dim].astype(np.float32)
    
    def extract_from_video(
        self, 
        video_path: str, 
        output_dir: str = 'video_features',
        sample_rate: int = 1,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract per-frame motion features using image subtraction.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save features
            sample_rate: Sample every Nth frame before differencing (1 = every frame)
            verbose: Whether to show progress
            
        Returns:
            Tuple of (features array, metadata dict)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = video_path.stem
        
        if verbose:
            print(f"\n🎬 Extracting video features from {video_name}")
            print(f"   Total frames: {total_frames:,}")
            print(f"   Sample rate: every {sample_rate} frame(s)")
            print(f"   Method: frame differencing (image subtraction)")
            print(f"   Expected output: ~{total_frames // sample_rate:,} feature vectors")
        
        # Process video
        features_list = []
        frame_number = 0
        prev_sampled_gray = None
        
        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frames", disable=not verbose)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_number % sample_rate == 0:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_sampled_gray is None:
                    feature_vec = np.zeros(self.feature_dim, dtype=np.float32)
                else:
                    feature_vec = self._extract_motion_feature(prev_sampled_gray, curr_gray)

                features_list.append(feature_vec)
                prev_sampled_gray = curr_gray
            
            frame_number += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        features = np.array(features_list)
        
        if verbose:
            print(f"✅ Extracted {len(features):,} feature vectors")
            print(f"   Feature shape: {features.shape}")
        
        # Save features
        features_path = output_dir / f"{video_name}_video_features.npy"
        np.save(features_path, features)
        
        if verbose:
            print(f"💾 Saved to: {features_path}")
        
        # Save metadata
        metadata = {
            'video_path': str(video_path.absolute()),
            'video_name': video_name,
            'method': 'frame_differencing',
            'feature_dim': self.feature_dim,
            'grid_shape': [self.grid_h, self.grid_w],
            'num_frames': len(features),
            'sample_rate': sample_rate,
            'extraction_timestamp': datetime.now().isoformat(),
            'video_metadata': {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps > 0 else 0
            }
        }
        
        metadata_path = output_dir / f"{video_name}_video_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return features, metadata


class VideoPreprocessor:
    """
    Combined preprocessor for extracting both pose and video features
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8l-pose.pt',
        device: str = 'cuda',
        feature_dim: int = 400,
        batch_size: int = 64
    ):
        """
        Initialize combined preprocessor
        
        Args:
            yolo_model_path: Path to YOLOv8-Pose model
            device: Device for computation ('cuda' or 'cpu')
            feature_dim: Dimension of video features
            batch_size: Batch size for video feature extraction
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu' and device == 'cuda':
            print("⚠️  CUDA not available, falling back to CPU")
        
        # Initialize extractors
        self.pose_extractor = YOLOPoseExtractor(
            model_path=yolo_model_path,
            device=self.device,
            batch_size=batch_size,
        )
        
        self.video_extractor = VideoFeatureExtractor(
            device=self.device,
            feature_dim=feature_dim,
            batch_size=batch_size
        )
    
    def preprocess_video(
        self,
        video_path: str,
        output_base_dir: str = '.',
        pose_conf_threshold: float = 0.3,
        video_sample_rate: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Preprocess a video by extracting both pose and video features
        
        Args:
            video_path: Path to input video file
            output_base_dir: Base directory for outputs
            pose_conf_threshold: Confidence threshold for pose detection
            video_sample_rate: Sample rate for video frames
            verbose: Whether to show progress
            
        Returns:
            Dictionary with paths and metadata for both outputs
        """
        video_path = Path(video_path)
        output_base_dir = Path(output_base_dir)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🎯 Preprocessing Video: {video_path.name}")
            print(f"{'='*80}\n")
        
        # Extract pose features
        pose_output_dir = output_base_dir / 'pose_features'
        pose_data, pose_metadata = self.pose_extractor.extract_from_video(
            video_path=str(video_path),
            output_dir=str(pose_output_dir),
            conf_threshold=pose_conf_threshold,
            verbose=verbose
        )
        
        # Extract video features
        video_output_dir = output_base_dir / 'video_features'
        video_features, video_metadata = self.video_extractor.extract_from_video(
            video_path=str(video_path),
            output_dir=str(video_output_dir),
            sample_rate=video_sample_rate,
            verbose=verbose
        )
        
        # Create summary
        summary = {
            'video_path': str(video_path.absolute()),
            'video_name': video_path.stem,
            'pose_features': {
                'output_path': str(pose_output_dir / f"{video_path.stem}_pose_features.json"),
                'total_frames': len(pose_data),
                'frames_with_poses': sum(1 for frame in pose_data if len(frame['poses']) > 0)
            },
            'video_features': {
                'output_path': str(video_output_dir / f"{video_path.stem}_video_features.npy"),
                'shape': video_features.shape,
                'feature_dim': self.video_extractor.feature_dim
            },
            'preprocessing_timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ Preprocessing Complete!")
            print(f"{'='*80}")
            print(f"\n📊 Summary:")
            print(f"   Pose features: {summary['pose_features']['output_path']}")
            print(f"   Video features: {summary['video_features']['output_path']}")
            print(f"   Total frames processed: {len(pose_data):,}")
            print(f"   Frames with poses: {summary['pose_features']['frames_with_poses']:,}")
            print(f"   Video feature shape: {summary['video_features']['shape']}")
            print(f"\n{'='*80}\n")
        
        return summary
    
    def preprocess_videos(
        self,
        video_paths: List[str],
        output_base_dir: str = '.',
        pose_conf_threshold: float = 0.3,
        video_sample_rate: int = 1,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Preprocess multiple videos
        
        Args:
            video_paths: List of paths to video files
            output_base_dir: Base directory for outputs
            pose_conf_threshold: Confidence threshold for pose detection
            video_sample_rate: Sample rate for video frames
            verbose: Whether to show progress
            
        Returns:
            List of summary dictionaries for each video
        """
        summaries = []
        
        for i, video_path in enumerate(video_paths, 1):
            if verbose:
                print(f"\n\n{'#'*80}")
                print(f"# Processing Video {i}/{len(video_paths)}")
                print(f"{'#'*80}\n")
            
            summary = self.preprocess_video(
                video_path=video_path,
                output_base_dir=output_base_dir,
                pose_conf_threshold=pose_conf_threshold,
                video_sample_rate=video_sample_rate,
                verbose=verbose
            )
            summaries.append(summary)
        
        if verbose:
            print(f"\n\n{'#'*80}")
            print(f"# All Videos Processed Successfully!")
            print(f"# Total: {len(video_paths)} videos")
            print(f"{'#'*80}\n")
        
        return summaries


def main():
    """Example usage of the preprocessing module"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess videos for pitch detection')
    parser.add_argument('videos', nargs='+', help='Path(s) to video file(s)')
    parser.add_argument('--output-dir', '-o', default='.', help='Output directory')
    parser.add_argument('--yolo-model', default='yolov8l-pose.pt', help='Path to YOLO model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--feature-dim', type=int, default=400, help='Video feature dimension')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for both pose and motion processing')
    parser.add_argument('--pose-conf', type=float, default=0.3, help='Pose detection confidence threshold')
    parser.add_argument('--sample-rate', type=int, default=1, help='Video frame sample rate')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        yolo_model_path=args.yolo_model,
        device=args.device,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size
    )
    
    # Process videos
    summaries = preprocessor.preprocess_videos(
        video_paths=args.videos,
        output_base_dir=args.output_dir,
        pose_conf_threshold=args.pose_conf,
        video_sample_rate=args.sample_rate,
        verbose=not args.quiet
    )
    
    # Save overall summary
    summary_path = Path(args.output_dir) / 'preprocessing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    if not args.quiet:
        print(f"📄 Overall summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
