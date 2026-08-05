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

## 🔄 End-to-End 2-Phase Training Pipeline (Recommended)

To handle domain discrepancy, a sequential two-phase training pipeline is established:
1. **Phase 1 (Pre-training)**: Train/fine-tune the base YOLO26 segment model on the **Business Card** dataset to learn generic card features and edges.
2. **Phase 2 (Fine-tuning)**: Fine-tune the best weights from Phase 1 on the **Vietnam ID Card (CCCD)** dataset.

The script `scripts/train_pipeline.py` handles:
* Automatically downloading and mapping both datasets from Roboflow if not already downloaded.
* Fixing paths in `data.yaml` files dynamically.
* Running Phase 1, saving its best weights, and feeding them as initial weights for Phase 2.
* Uploading separate runs to Weights & Biases (W&B) for both phases.

### CLI Arguments for Pipeline

| Argument               | Default            | Description                                                     |
| :--------------------- | :----------------- | :-------------------------------------------------------------- |
| `-m`, `--model`        | `yolo26n-seg.pt`   | Path or name of starting weights                                |
| `-b`, `--batch`        | `16`               | Batch size for training                                         |
| `--epochs_phase1`      | `50`               | Number of epochs for Phase 1 (Business Card)                    |
| `--epochs_phase2`      | `100`              | Number of epochs for Phase 2 (Vietnam ID Card)                  |
| `-p`, `--patience`     | `30`               | Early stopping patience                                         |
| `--cos_lr`             | *None*             | Use cosine learning rate scheduler                              |
| `--optimizer`          | `AdamW`            | Optimizer (`Adam`, `AdamW`, `SGD`, `RMSProp`, `auto`)           |
| `--skip_download`      | *None*             | Skip dataset downloading if folders exist                       |
| `--test`               | *None*             | Runs a quick end-to-end dry-run (1 epoch, 1 image)              |

### Examples

#### 1. Run Pipeline Test (Recommended to verify environment)
```bash
python scripts/train_pipeline.py --test
```

#### 2. Run Full 2-Phase Training
```bash
python scripts/train_pipeline.py -m yolo26n-seg.pt -b 32 --epochs_phase1 50 --epochs_phase2 100 --cos_lr
```

---

## 💡 Running Only Phase 2 (Fine-tuning again)

If your **Phase 1 (Pre-training)** achieved good results, but you want to fine-tune **Phase 2 (ID Card)** again with different settings (e.g., higher epochs, lower learning rate, or different batch size), you **do not need to re-run the entire pipeline**.

You can run `scripts/train_yolo26_segment.py` directly, passing the best weights of Phase 1 to the `--model` argument:

```bash
python scripts/train_yolo26_segment.py \
  --model model/segmentation/yolo26_business_card_pretrain/weights/best.pt \
  --data data/id_cards_dataset/data.yaml \
  --name yolo26_id_card_finetune_v2 \
  --project ID_Card_VN \
  --epochs 100 \
  --batch 32 \
  --cos_lr
```

---

## 📊 Legacy / Alternative: Single-Phase Training

If you need to train on the legacy single merged dataset (`data/merged_roboflow_dataset`):

### 1. Download Legacy Dataset
```bash
python scripts/download_dataset.py
```

### 2. Run Single-Phase Training
```bash
python scripts/train_yolo26_segment.py \
  --model yolo26n-seg.pt \
  --data data/merged_roboflow_dataset/data.yaml \
  --name yolo26_id_card_seg \
  --epochs 100 \
  --batch 16
```

---

## 📊 Outputs & Monitoring

1. **W&B Live Dashboard**:
   - Metrics are uploaded in real-time. Only essential charts (losses, mAPs, precision, recall, learning rate, F1 score) are logged to minimize noise.
   - Standard console logs are captured and synced to the W&B **Logs** tab automatically.
   - At the end of training, the model's best weights (`best.pt`) and evaluation charts (e.g. `BoxF1_curve.png`, `MaskF1_curve.png`, etc.) are automatically uploaded directly to the W&B Run Files and Media workspace.
2. **Local Output**:
   - Model weights and plots are saved locally under `model/segmentation/<run_name>/`.
   - Pretrained/Best weights are located at `model/segmentation/<run_name>/weights/best.pt`.
