# Change-Impact Graph Analysis - YOLO Segmentation Fine-tuning Collapse Fix

## Proposed Changes
We will modify `scripts/train_yolo26_segment.py` to add `optimizer="AdamW"` to `model.train()` calls (specifically Stage 2) to ensure the learning rate custom values (`lr0=0.0002`) are not ignored.

## Impact Graph
- **Direct Impacted Symbols**: `model.train()` calls in `scripts/train_yolo26_segment.py`.
- **Indirect Impacted Behaviors**:
  - In Stage 2: YOLO will initialize the AdamW optimizer with our specified learning rate `lr0=0.0002` instead of overriding it to `0.002`.
  - The model weights will not collapse, and training will continue from the high mAP50 warm-up base.
- **Affected User Flows**:
  - CLI execution: `python scripts/train_yolo26_segment.py --staged`
  - Wandb logging: epoch logging will remain continuous and stable.
