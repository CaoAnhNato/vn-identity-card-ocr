# Evidence Report - YOLO Segmentation Fine-tuning Collapse Fix

## Requirement Contract Reference
- Requirement Contract: [requirement.md](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/specs/task_yolo_seg_collapse/requirement.md)
- Acceptance Criteria verified: `AC-01`, `AC-02`, `AC-03`, `AC-04`.

## Change Impact Graph Reference
- Impact analysis: [impact_analysis.md](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/specs/task_yolo_seg_collapse/impact_analysis.md)

## Verification Evidence

### 1. Acceptance Unit/Integration Test
- Executed Test case: [test_req_yolo_seg.py](file:///c:/Users/Admin/HUIT - Học Tập/Năm 3/DocU/tests/acceptance/test_req_yolo_seg.py)
- Command: `python tests/acceptance/test_req_yolo_seg.py`
- Result: Passed successfully. Verified that when `optimizer="AdamW"` is passed, `lr0` is set to `0.0002` and NOT overridden to `0.002` by YOLO.
```python
optimizer: AdamW(lr=0.0002, momentum=0.937) with parameter groups 134 weight(decay=0.0)...
Ran 1 test in 7.738s
OK
```

### 2. End-to-End Staged Training Test
- Command: `python scripts/train_yolo26_segment.py --test --staged`
- Result: Completed successfully.
- Log snippet from W&B run summary:
```yaml
Run summary:
             epoch 3
            lr/pg0 0.0002
            lr/pg1 0.0002
            lr/pg2 0.0002
```
This confirms that the learning rate is correctly set to `0.0002` in Stage 2 fine-tuning!

## Human Acceptance Checklist
- [x] AC-01: Explicit optimizer is passed to Stage 2 training.
- [x] AC-02: Fine-tuning learning rate is respected and not overridden.
- [x] AC-03: Staged training runs successfully end-to-end.
- [x] AC-04: Test mode runs successfully without exceptions.
