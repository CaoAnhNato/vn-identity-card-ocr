# Change-Impact Graph Analysis - YOLO Segmentation Fine-tuning Collapse and Instability Fix

## Proposed Changes
We will modify `scripts/train_yolo26_segment.py` to:
1. Fix `SafeCoarseDropout` to override `apply_to_mask` and `apply_to_masks` to return unmodified masks.
2. Determine `lr0` based on optimizer choice in Stage 1 and Standard training: default `lr0 = 0.002` if optimizer is `Adam` or `AdamW`, and `lr0 = 0.01` if `SGD` or `auto`.
3. In Stage 2 training, set `warmup_epochs=0.0` to disable warmup and prevent bias learning rate spikes.

## Impact Graph
- **Direct Impacted Symbols**:
  - `SafeCoarseDropout` class in `scripts/train_yolo26_segment.py`
  - `model.train()` parameter configs in `scripts/train_yolo26_segment.py`
- **Indirect Impacted Behaviors**:
  - Training images will have occlusion blocks, but ground-truth masks remain solid. This improves card segmentation completeness.
  - Stage 1 uses a stable learning rate for AdamW, yielding high baseline mAP.
  - Stage 2 transitions without warmup, preserving pre-trained weights/biases and avoiding representation collapse.
- **Affected User Flows**:
  - Staged training via command line options.
