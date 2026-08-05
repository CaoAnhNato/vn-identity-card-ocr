# YOLO Instance Segmentation Training Guide

This guide describes how to set up the environment, download datasets, and train the YOLO26 instance segmentation model on a GPU container (Docker, runpod, vast.ai, etc.).

---

## 🛠 Prerequisites

Ensure you have a GPU container with:

- **CUDA** (v11.8 or v12.x recommended)
- **Python** (3.8 - 3.12)
- **Git**

---

## 🚀 Setup Steps

### Step 1: Install Dependencies

Run the following command to install only the training and dataset downloading packages:

```bash
pip install -r requirements_train.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root directory and add your Weights & Biases API key:

```bash
echo "WANDB_API_KEY=your_wandb_api_key_here" > .env
```

*(Note: W&B is used to log training loss, learning rate, precision, recall, mAP, and F1 scores).*

### Step 3: Estimate GPU VRAM & Optimize Batch Size (Optional)

Before starting full training, you can run the VRAM estimation script to determine the peak GPU memory usage and choose the optimal batch size for your GPU container:

```bash
python scripts/estimate_vram.py --model yolo26n-seg.pt --imgsz 640
```

This script will output the base model size, peak allocated memory, and recommended batch sizes.

---

## 🔄 Joint Training on Merged Dataset (Highly Recommended)

Training on both **Business Cards** and **ID Cards** simultaneously (Joint Training) is the most robust strategy. This prevents **Scale Shift Bias** (Business Cards occupy 55% area while ID Cards occupy 27%) and **Gradient Interference** that occurs when fine-tuning sequentially.

### Step 1: Download & Merge Datasets

Run the download script to fetch and combine the datasets into `data/merged_roboflow_dataset`:
```bash
python scripts/download_dataset.py
```

### Step 2: Run Joint Training

Train the model using the optimized hyperparameters (Medium model, disabled custom Albumentations, default YOLO spatial scaling):

```bash
python scripts/train_yolo26_segment.py \
  --model yolo26m-seg.pt \
  --data data/merged_roboflow_dataset/data.yaml \
  --name yolo26m_joint_training \
  --epochs 150 \
  --batch 16 \
  --no_custom_aug \
  --mosaic 1.0 \
  --degrees 0.0 \
  --scale 0.5 \
  --translate 0.1 \
  --perspective 0.0
```

*Note: The `--test` flag can be appended to run a 1-epoch dry-run test on a single image to verify your setup.*

---

## 🛠️ CLI Arguments for `train_yolo26_segment.py`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model` | `yolo26n-seg.pt` | Path or name of YOLO model weights (e.g. `yolo26m-seg.pt`). |
| `-b`, `--batch` | `16` | Batch size for training. |
| `-e`, `--epochs` | `100` | Number of training epochs (100–150 recommended for joint training). |
| `-n`, `--name` | `yolo26_id_card_seg` | Name of the training run / model saved to W&B. |
| `-d`, `--data` | *None* | Path to dataset `data.yaml` configuration. |
| `--project` | `ID_Card_VN` | W&B project name to log metrics under. |
| `-p`, `--patience` | `50` | Early stopping patience (epochs of no improvement before stopping training). |
| `--cos_lr` | *False* | Flag to use cosine learning rate scheduler during training. |
| `--optimizer` | `AdamW` | Optimizer to use (`Adam`, `AdamW`, `SGD`, `RMSProp`, `auto`). |
| `--freeze` | *None* | Number of initial layers to freeze (e.g., `10` to freeze the backbone). |
| `--staged` | *False* | Flag to enable staged training: freeze backbone first, then unfreeze and fine-tune. |
| `--freeze_epochs` | `30` | Number of epochs to train with frozen backbone in staged mode. |
| `--no_custom_aug` | *False* | Flag to disable custom Albumentations (falls back to default YOLOv8/v11 augmentations). |
| `--mosaic` | `0.2` | Mosaic augmentation probability (set to `1.0` or `0.5` for joint training). |
| `--degrees` | `0.0` | Rotation angle in degrees (keep at `0.0` to prevent bounding box phình to). |
| `--scale` | `0.5` | Scale factor for zoom augmentation (keep at `0.5` to bridge 2x card scale difference). |
| `--translate` | `0.1` | Translation factor. |
| `--perspective` | `0.0` | Perspective transformation factor. |
| `--hsv_h` | `0.015` | HSV Hue gain. |
| `--hsv_s` | `0.7` | HSV Saturation gain. |
| `--hsv_v` | `0.4` | HSV Value gain. |
| `--test` | *False* | Flag to run a quick training test with exactly 1 image and 1 epoch. |

---

## 📊 Legacy / Alternative: 2-Phase Training Pipeline

If you still need to run sequential pre-training on Business Cards followed by fine-tuning on ID Cards, you can run:

```bash
python scripts/train_pipeline.py -m yolo26m-seg.pt -b 16 --epochs_phase1 50 --epochs_phase2 100
```
*Caution: Sequential training can lead to representation collapse and scale bias on the second phase due to the drastic difference in card sizes.*


---

## 📊 Outputs & Monitoring

1. **W&B Live Dashboard**:
   - Metrics are uploaded in real-time. Only essential charts (losses, mAPs, precision, recall, learning rate, F1 score) are logged to minimize noise.
   - Standard console logs are captured and synced to the W&B **Logs** tab automatically.
   - At the end of training, the model's best weights (`best.pt`) and evaluation charts (e.g. `BoxF1_curve.png`, `MaskF1_curve.png`, etc.) are automatically uploaded directly to the W&B Run Files and Media workspace.
2. **Local Output**:
   - Model weights and plots are saved locally under `model/segmentation/<run_name>/`.
   - Pretrained/Best weights are located at `model/segmentation/<run_name>/weights/best.pt`.
