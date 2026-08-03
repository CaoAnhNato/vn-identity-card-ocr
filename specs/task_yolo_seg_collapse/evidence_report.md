# Evidence Report - YOLO Segmentation Fine-tuning Collapse and Instability Fix

## Requirement Contract Reference
- Requirement Contract: [requirement.md](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/specs/task_yolo_seg_collapse/requirement.md)
- Acceptance Criteria verified: `AC-01`, `AC-02`, `AC-03`, `AC-04`.

## Change Impact Graph Reference
- Impact analysis: [impact_analysis.md](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/specs/task_yolo_seg_collapse/impact_analysis.md)

## Verification Evidence

### 1. Acceptance Unit/Integration Test
- Executed Test case: [test_req_yolo_seg.py](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/tests/acceptance/test_req_yolo_seg.py)
- Command: `python tests/acceptance/test_req_yolo_seg.py`
- Result: Passed successfully. Verified that `SafeCoarseDropout` applies cutout holes to the image pixels, but leaves bounding boxes and target masks fully intact.
```
.
----------------------------------------------------------------------
Ran 1 test in 0.003s
OK
```

### 2. End-to-End Staged Training Test
- Command: `python scripts/train_yolo26_segment.py --test --staged`
- Result: Completed successfully.
- Verification details:
  - Stage 1 uses `lr0 = 0.002` for AdamW (preventing `0.01` instability).
  - Stage 2 uses `lr0 = 0.0002` and `warmup_epochs = 0.0` (preventing warmup bias LR spike to `0.1` and preserving pre-trained weights/biases).
  - Augmentations preserve target masks and bounding boxes.

## Human Acceptance Checklist
- [x] AC-01: Correct learning rate (0.002) is set for Stage 1/Standard AdamW optimizer.
- [x] AC-02: Warmup is disabled (`warmup_epochs=0.0`) in Stage 2 to prevent bias LR spikes.
- [x] AC-03: `SafeCoarseDropout` does not modify target masks.
- [x] AC-04: Test mode runs successfully end-to-end without exceptions.
