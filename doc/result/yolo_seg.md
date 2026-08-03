**Run**

```bash
python scripts/train_yolo26_segment.py -m yolo26n-seg.pt -b 64 -e 100 -p 20 --staged --freeze_epochs 30 --cos_lr -n yolo26_cccd_staged
```
