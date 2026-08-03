# Requirement Contract - YOLO Segmentation Fine-tuning Collapse Fix

## Problem Description
During Staged Training (`--staged`), Stage 2 (Fine-tuning) collapses immediately. 
Validation mAP50(B) drops from `0.687` to `0.142`, and validation mAP50(M) drops from `0.704` to `0.023` within a few epochs.
EarlyStopping triggers prematurely.

## Root Cause
YOLO's optimizer setting `optimizer='auto'` ignores custom `lr0` and `momentum` values. It defaults to AdamW with learning rate `lr0=0.002` (which is 10x higher than the fine-tuning rate of `0.0002`). This extremely high learning rate on an unfrozen backbone causes catastrophic forgetting (representation collapse).

## Acceptance Criteria
- **AC-01**: In Stage 2 of `--staged` training, the optimizer must be explicitly set to `AdamW` (or custom user-selected optimizer) instead of `auto`.
- **AC-02**: The custom fine-tuning learning rate `lr0=0.0002` must be successfully respected by YOLO during Stage 2 (verified via logs/training configurations).
- **AC-03**: In Stage 1 of `--staged` training and Standard Training, the optimizer can be specified explicitly or kept consistent.
- **AC-04**: The `--staged` training script must continue to execute successfully under test mode (`--test`) without raising any exceptions.
