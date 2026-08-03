# Requirement Contract - YOLO Segmentation Fine-tuning Collapse and Instability Fix

## Problem Description
Despite forcing the optimizer to `AdamW` in Stage 2, the training still collapses:
1. **Stage 1 Instability**: `Box mAP50` was stuck at `0.35` (down from `0.68` in the first run). This is because forcing `optimizer='AdamW'` without setting `lr0` defaults to `lr0=0.01` (which is meant for SGD and is 5-10x too high for AdamW).
2. **Stage 2 Collapse**: Even with `lr0=0.0002`, `Mask mAP50` collapsed immediately to `0.018` in Stage 2. This is because YOLO triggered a 3-epoch warmup, during which the bias learning rate spiked to `0.1` (default `warmup_bias_lr`), destroying the trained head biases.
3. **Occlusion Augmentations Issue**: The `CoarseDropout` augmentation was dropping pixels from the ground-truth target mask, teaching the model to predict card masks with holes, which degrades validation performance.

## Root Causes
1. **Optimizer-LR Mismatch**: Using `AdamW` with `lr0=0.01` in Stage 1.
2. **Warmup Shock**: The 3-epoch warmup in Stage 2 setting bias learning rate to `0.1`.
3. **Mask Destruction**: `CoarseDropout` applying holes to the ground-truth mask.

## Acceptance Criteria
- **AC-01**: In Stage 1 (and standard training), if the optimizer is set to `Adam` or `AdamW`, the learning rate `lr0` must be set to `0.002` (consistent with YOLO's auto-tuned rate for AdamW) instead of `0.01`.
- **AC-02**: In Stage 2, `warmup_epochs` must be explicitly set to `0.0` to disable the warmup phase and prevent bias learning rate spikes.
- **AC-03**: In `SafeCoarseDropout`, the target mask must remain fully solid (unmodified) by overriding `apply_to_mask` and `apply_to_masks` to return the inputs unmodified.
- **AC-04**: The staged training script must execute successfully under test mode (`--test`) without raising exceptions.
