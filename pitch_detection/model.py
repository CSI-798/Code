"""
Neural network model architecture for pitch detection.
"""

import torch
import torch.nn as nn


class MultiModalBMN(nn.Module):
    """Multi-Modal Boundary Matching Network for pitch detection"""
    
    def __init__(self, pose_dim=68, video_dim=400, temporal_scale=100, hidden_dim=256):
        super().__init__()
        self.temporal_scale = temporal_scale
        self.hidden_dim = hidden_dim
        
        # Enhanced feature processing for better IoU performance
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Slightly higher dropout for regularization
            nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm1d
        )
        
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Slightly higher dropout for regularization
            nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm1d
        )
        
        # Temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Sequence modeling over ordered frames
        self.temporal_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Multi-scale temporal pyramid for variable action durations
        self.temporal_pyramid = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
        ])
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Prediction heads
        self.frame_predictor = nn.Linear(hidden_dim, 1)
        self.start_predictor = nn.Linear(hidden_dim, 1)
        self.end_predictor = nn.Linear(hidden_dim, 1)
        
        # Attention for multi-modal fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, pose_features, video_features):
        batch_size, seq_len = pose_features.shape[:2]
        
        # Encode features
        pose_encoded = self.pose_encoder(pose_features)
        video_encoded = self.video_encoder(video_features)
        
        # Multi-modal attention
        combined = torch.cat([pose_encoded, video_encoded], dim=-1)
        attention_weights = self.attention(combined)
        
        # Apply attention
        weighted_pose = pose_encoded * attention_weights[:, :, 0:1]
        weighted_video = video_encoded * attention_weights[:, :, 1:2]
        
        # Combine features
        fused_features = torch.cat([weighted_pose, weighted_video], dim=-1)
        
        # Temporal convolution (batch_size, seq_len, features) -> (batch_size, features, seq_len)
        temporal_input = fused_features.transpose(1, 2)
        temporal_output = self.temporal_conv(temporal_input)
        temporal_features = temporal_output.transpose(1, 2)
        
        # Sequence-aware temporal modeling (captures longer action dynamics)
        rnn_features, _ = self.temporal_rnn(temporal_features)
        temporal_features = self.rnn_projection(rnn_features)

        # Multi-scale temporal fusion (improves TAD boundary quality)
        pyramid_input = temporal_features.transpose(1, 2)
        pyramid_features = [torch.relu(branch(pyramid_input)) for branch in self.temporal_pyramid]
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        temporal_features = self.temporal_fusion(pyramid_concat).transpose(1, 2)

        # Predictions (return logits for training, apply sigmoid during inference)
        frame_logits = self.frame_predictor(temporal_features).squeeze(-1)
        start_logits = self.start_predictor(temporal_features).squeeze(-1)
        end_logits = self.end_predictor(temporal_features).squeeze(-1)
        
        # Apply sigmoid for inference (when not training)
        if not self.training:
            frame_predictions = torch.sigmoid(frame_logits)
            start_predictions = torch.sigmoid(start_logits)
            end_predictions = torch.sigmoid(end_logits)
        else:
            frame_predictions = frame_logits
            start_predictions = start_logits
            end_predictions = end_logits
        
        return {
            'frame_predictions': frame_predictions,
            'start_predictions': start_predictions,
            'end_predictions': end_predictions,
            'frame_logits': frame_logits,
            'start_logits': start_logits,
            'end_logits': end_logits,
            'temporal_features': temporal_features,
            'attention_weights': attention_weights,
        }
