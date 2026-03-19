"""
Dataset and data loading utilities for pitch detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import Tuple


class GPUSlidingWindowDataset(Dataset):
    """GPU-optimized sliding window dataset for pose and video features"""
    
    def __init__(
        self,
        pose_files,
        video_files,
        label_files,
        window_size=100,
        stride=50,
        fps=30,
        augment=False,
        augment_factor=1,
        temporal_shift=0,
        pose_noise_std=0.0,
        video_noise_std=0.0,
        feature_dropout_prob=0.0,
    ):
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self.augment = augment
        self.augment_factor = max(1, int(augment_factor))
        self.temporal_shift = max(0, int(temporal_shift))
        self.pose_noise_std = max(0.0, float(pose_noise_std))
        self.video_noise_std = max(0.0, float(video_noise_std))
        self.feature_dropout_prob = min(1.0, max(0.0, float(feature_dropout_prob)))
        
        # Load all data
        self.pose_data = []
        self.video_data = []
        self.labels = []
        
        for i, pose_file in enumerate(pose_files):
            # Load pose features with proper error handling
            with open(pose_file, 'r') as f:
                pose_json = json.load(f)
            
            # Process poses frame by frame to handle missing poses
            poses_list = []
            for frame in pose_json['poses']:
                if frame['poses'] and len(frame['poses']) > 0:
                    poses = frame['poses']
                    primary_pose_index = frame.get('primary_pose_index', None)

                    if isinstance(primary_pose_index, int) and 0 <= primary_pose_index < len(poses):
                        selected_pose = poses[primary_pose_index]
                    else:
                        selected_pose = max(poses, key=lambda pose: pose.get('pose_score', 0.0))

                    keypoints = selected_pose['keypoints']
                    if len(keypoints) == 34:  # Ensure we have x,y for 17 joints
                        poses_list.append(keypoints)
                    else:
                        poses_list.append(np.zeros(34))  # Fallback for malformed data
                else:
                    poses_list.append(np.zeros(34))  # No pose detected
            
            # Convert to numpy array safely
            poses = np.array(poses_list)
            
            # Extend to 68 features (add confidence and visibility)
            extended_poses = np.zeros((poses.shape[0], 68))
            extended_poses[:, :34] = poses
            extended_poses[:, 34:51] = 0.8  # Default confidence for 17 joints
            extended_poses[:, 51:68] = 1.0  # Default visibility for 17 joints
            
            # Load video features
            video_features = np.load(video_files[i] if i < len(video_files) else video_files[0])
            
            # DEBUG: Print loaded data lengths
            print(f"  DEBUG - Video {i+1}: Loaded {len(extended_poses)} pose frames, {len(video_features)} video frames")
            
            # Load and parse labels correctly
            with open(label_files[i] if i < len(label_files) else label_files[0], 'r') as f:
                raw_labels = json.load(f)
            
            # Parse the label format correctly
            pitch_segments = []
            if isinstance(raw_labels, list) and len(raw_labels) > 0:
                # Handle the actual label format: [{"videoLabels": [{"ranges": [...]}]}]
                video_data = raw_labels[0]  # First video in the list
                if 'videoLabels' in video_data:
                    for label_group in video_data['videoLabels']:
                        if 'ranges' in label_group and 'timelinelabels' in label_group:
                            # Only process "Pitch" labels
                            if 'Pitch' in label_group['timelinelabels']:
                                for range_data in label_group['ranges']:
                                    pitch_segments.append({
                                        'start_frame': range_data['start'],
                                        'end_frame': range_data['end']
                                    })
            
            # Create standardized label format
            label_data = {'segments': pitch_segments}
            
            print(f"  Video {i+1}: Found {len(pitch_segments)} pitch segments")
            for j, seg in enumerate(pitch_segments):
                duration = (seg['end_frame'] - seg['start_frame']) / fps
                print(f"    Pitch {j+1}: frames {seg['start_frame']}-{seg['end_frame']} ({duration:.2f}s)")
            
            # Sync lengths and ensure minimum window size
            min_len = min(len(extended_poses), len(video_features))
            if min_len < window_size:
                print(f"Warning: Video {i} has only {min_len} frames, less than window size {window_size}")
                # Pad with zeros if too short
                pad_frames = window_size - min_len
                extended_poses = np.pad(extended_poses, ((0, pad_frames), (0, 0)), mode='constant')
                video_features = np.pad(video_features, ((0, pad_frames), (0, 0)), mode='constant')
                min_len = len(extended_poses)  # Update min_len to padded length
            
            # Use synchronized length (don't truncate unnecessarily)
            extended_poses = extended_poses[:min_len]
            video_features = video_features[:min_len]
            
            self.pose_data.append(extended_poses)
            self.video_data.append(video_features)
            self.labels.append(label_data)
        
        # Create sliding windows
        self.windows = []
        for video_idx in range(len(self.pose_data)):
            num_frames = len(self.pose_data[video_idx])
            for start in range(0, max(1, num_frames - window_size + 1), stride):
                end = start + window_size
                if end <= num_frames:  # Ensure we don't exceed bounds
                    self.windows.append({
                        'video_idx': video_idx,
                        'start_frame': start,
                        'end_frame': end
                    })
        
        print(f"Dataset created: {len(self.windows)} windows from {len(pose_files)} videos")
        if len(self.windows) == 0:
            raise ValueError("No valid windows created. Check your data files and window size.")

        if self.augment:
            print(
                f"Augmentation enabled: factor={self.augment_factor}, shift={self.temporal_shift}, "
                f"pose_noise={self.pose_noise_std}, video_noise={self.video_noise_std}, "
                f"dropout={self.feature_dropout_prob}"
            )

    def _temporal_shift_arrays(
        self,
        shift: int,
        pose_window: np.ndarray,
        video_window: np.ndarray,
        frame_labels: np.ndarray,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shift all time-aligned arrays together and zero-pad introduced edges."""
        if shift == 0:
            return pose_window, video_window, frame_labels, start_labels, end_labels

        pose_shifted = np.roll(pose_window, shift=shift, axis=0)
        video_shifted = np.roll(video_window, shift=shift, axis=0)
        frame_shifted = np.roll(frame_labels, shift=shift, axis=0)
        start_shifted = np.roll(start_labels, shift=shift, axis=0)
        end_shifted = np.roll(end_labels, shift=shift, axis=0)

        if shift > 0:
            pose_shifted[:shift] = 0
            video_shifted[:shift] = 0
            frame_shifted[:shift] = 0
            start_shifted[:shift] = 0
            end_shifted[:shift] = 0
        else:
            pose_shifted[shift:] = 0
            video_shifted[shift:] = 0
            frame_shifted[shift:] = 0
            start_shifted[shift:] = 0
            end_shifted[shift:] = 0

        return pose_shifted, video_shifted, frame_shifted, start_shifted, end_shifted

    def _augment_window(
        self,
        pose_window: np.ndarray,
        video_window: np.ndarray,
        frame_labels: np.ndarray,
        start_labels: np.ndarray,
        end_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply stochastic augmentation to one training window."""
        aug_pose = pose_window.copy()
        aug_video = video_window.copy()
        aug_frame = frame_labels.copy()
        aug_start = start_labels.copy()
        aug_end = end_labels.copy()

        if self.temporal_shift > 0:
            shift = np.random.randint(-self.temporal_shift, self.temporal_shift + 1)
            aug_pose, aug_video, aug_frame, aug_start, aug_end = self._temporal_shift_arrays(
                shift,
                aug_pose,
                aug_video,
                aug_frame,
                aug_start,
                aug_end,
            )

        if self.pose_noise_std > 0:
            aug_pose += np.random.normal(0.0, self.pose_noise_std, size=aug_pose.shape)

        if self.video_noise_std > 0:
            aug_video += np.random.normal(0.0, self.video_noise_std, size=aug_video.shape)

        if self.feature_dropout_prob > 0:
            keep_mask = (np.random.rand(aug_pose.shape[0], 1) > self.feature_dropout_prob).astype(np.float32)
            aug_pose *= keep_mask
            aug_video *= keep_mask

        return aug_pose, aug_video, aug_frame, aug_start, aug_end
    
    def __len__(self):
        if self.augment:
            return len(self.windows) * self.augment_factor
        return len(self.windows)
    
    def __getitem__(self, idx):
        base_idx = idx % len(self.windows)
        augment_replica = idx // len(self.windows)
        window = self.windows[base_idx]
        video_idx = window['video_idx']
        start = window['start_frame']
        end = window['end_frame']
        
        # Extract window data
        pose_window = self.pose_data[video_idx][start:end]
        video_window = self.video_data[video_idx][start:end]
        
        # Ensure correct window size
        if len(pose_window) != self.window_size:
            # Pad if necessary
            pad_frames = self.window_size - len(pose_window)
            pose_window = np.pad(pose_window, ((0, pad_frames), (0, 0)), mode='constant')
            video_window = np.pad(video_window, ((0, pad_frames), (0, 0)), mode='constant')
        
        # Create frame labels based on pitch segments
        frame_labels = np.zeros(self.window_size)
        start_labels = np.zeros(self.window_size)
        end_labels = np.zeros(self.window_size)
        
        # Add pitch segments from labels
        if 'segments' in self.labels[video_idx]:
            for segment in self.labels[video_idx]['segments']:
                # Calculate overlap with current window
                seg_start = segment['start_frame']
                seg_end = segment['end_frame']
                
                # Find intersection with current window
                window_seg_start = max(0, seg_start - start)
                window_seg_end = min(self.window_size, seg_end - start)
                
                # Only add if there's actual overlap
                if window_seg_start < self.window_size and window_seg_end > 0:
                    # Ensure we don't go out of bounds
                    window_seg_start = max(0, window_seg_start)
                    window_seg_end = min(self.window_size, window_seg_end)
                    
                    # Mark frames as pitch
                    frame_labels[window_seg_start:window_seg_end] = 1.0
                    
                    # Mark start boundary
                    if window_seg_start < self.window_size:
                        start_labels[window_seg_start] = 1.0
                    
                    # Mark end boundary  
                    if window_seg_end > 0 and window_seg_end <= self.window_size:
                        end_labels[window_seg_end-1] = 1.0

        should_augment = self.augment and augment_replica > 0
        if should_augment:
            pose_window, video_window, frame_labels, start_labels, end_labels = self._augment_window(
                pose_window,
                video_window,
                frame_labels,
                start_labels,
                end_labels,
            )
        
        return {
            'pose_features': torch.FloatTensor(pose_window),
            'video_features': torch.FloatTensor(video_window),
            'frame_labels': torch.FloatTensor(frame_labels),
            'start_boundary_labels': torch.FloatTensor(start_labels),
            'end_boundary_labels': torch.FloatTensor(end_labels),
            'window_info': window
        }


def create_dataloader(dataset, batch_size=32, num_workers=2, shuffle=True):
    """Create optimized dataloader"""
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
