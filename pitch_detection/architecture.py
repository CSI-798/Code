"""
Visualization of the modularized pitch detection framework architecture.
"""

ARCHITECTURE_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────┐
│                    PITCH DETECTION FRAMEWORK                            │
│                         Modular Architecture                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────┤
│  config.py                                                              │
│  ├── MODEL_CONFIG        (architecture parameters)                      │
│  ├── TRAINING_CONFIG     (training hyperparameters)                     │
│  ├── LOSS_CONFIG         (loss function weights)                        │
│  ├── DATASET_CONFIG      (data loading settings)                        │
│  ├── INFERENCE_CONFIG    (prediction settings)                          │
│  └── EVALUATION_CONFIG   (evaluation metrics)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE MODULES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          │
│  │   model.py    │    │  dataset.py   │    │    loss.py    │          │
│  ├───────────────┤    ├───────────────┤    ├───────────────┤          │
│  │ MultiModalBMN │    │ Sliding       │    │ IoU Loss      │          │
│  │ - pose encoder│    │ Window        │    │ Dice Loss     │          │
│  │ - video enc.  │    │ Dataset       │    │ Precision     │          │
│  │ - attention   │    │               │    │ Tversky       │          │
│  │ - temporal    │    │ DataLoader    │    │ FP Penalty    │          │
│  │ - predictors  │    │               │    │ Focal Loss    │          │
│  └───────────────┘    └───────────────┘    └───────────────┘          │
│          │                     │                     │                 │
│          └─────────────────────┴─────────────────────┘                 │
│                                │                                        │
└────────────────────────────────┼────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRAINING & INFERENCE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────┐    ┌──────────────────────────┐         │
│  │      trainer.py           │    │     inference.py         │         │
│  ├───────────────────────────┤    ├──────────────────────────┤         │
│  │ train_pitch_detector()    │    │ load_best_model()        │         │
│  │ - Create dataset          │    │ find_best_checkpoint()   │         │
│  │ - Initialize model        │    │ predict_pitch_timestamps()│         │
│  │ - Training loop           │    │ evaluate_model()         │         │
│  │ - Checkpoint saving       │    │                          │         │
│  │ - Metric tracking         │    │                          │         │
│  └───────────────────────────┘    └──────────────────────────┘         │
│                  │                              │                       │
│                  └──────────────┬───────────────┘                       │
│                                 ▼                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        METRICS & EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│  metrics.py                                                             │
│  ├── calculate_segment_iou()                                            │
│  ├── extract_segments_from_predictions()                                │
│  ├── evaluate_pitch_detection()                                         │
│  └── analyze_dataset_distribution()                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Files                  Processing              Output            │
│  ═══════════                  ══════════              ══════            │
│                                                                         │
│  pose.json    ─┐                                                        │
│               ├──► Dataset ──► Model ──► Loss ──► Train ──► Checkpoint │
│  video.npy    ─┤                                                        │
│               │                                                         │
│  labels.json ─┘                                                         │
│                                                                         │
│                                                                         │
│  Checkpoint ──────► Load Model ──► Predict ──► Segments                │
│                                                                         │
│  new_pose.json  ─┐                                                      │
│                 ├──► Predict ──► Segments + Timestamps                  │
│  new_video.npy ─┘                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      USAGE PATTERNS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. TRAINING                                                            │
│     from pitch_detection import train_pitch_detector                   │
│     model, history, ckpt = train_pitch_detector(...)                   │
│                                                                         │
│  2. INFERENCE                                                           │
│     from pitch_detection import load_best_model, predict_pitch_timestamps│
│     model, info = load_best_model(checkpoint_path)                     │
│     segments, preds = predict_pitch_timestamps(model, ...)             │
│                                                                         │
│  3. EVALUATION                                                          │
│     from pitch_detection import evaluate_model_performance             │
│     results = evaluate_model_performance(model, ...)                   │
│                                                                         │
│  4. CUSTOM CONFIGURATION                                                │
│     from pitch_detection import TRAINING_CONFIG, print_config          │
│     print_config()                                                      │
│     TRAINING_CONFIG['num_epochs'] = 100                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    MODULAR BENEFITS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ✓ Easy to modify individual components                                │
│  ✓ Clear separation of concerns                                        │
│  ✓ Reusable across projects                                            │
│  ✓ Simple to test                                                       │
│  ✓ Better code organization                                            │
│  ✓ Centralized configuration                                           │
│  ✓ Independent module development                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""

def print_architecture():
    """Print the architecture diagram"""
    print(ARCHITECTURE_DIAGRAM)

if __name__ == "__main__":
    print_architecture()
