# YOLO Instance Segmentation Training Guide

This guide describes how to set up the environment, download the dataset, and train the YOLO26 instance segmentation model on a GPU container (Docker, runpod, vast.ai, etc.).

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

### Step 2: Configure environment Variables
Create a `.env` file in the project root directory and add your Weights & Biases API key:
```bash
echo "WANDB_API_KEY=your_wandb_api_key_here" > .env
```
*(Note: W&B is used to log training loss, learning rate, precision, recall, mAP, and F1 scores).*

### Step 3: Download the Dataset
The raw dataset is hosted on Roboflow. Run the download script to automatically download it into `data/raw/ID-card-1`:
```bash
python scripts/download_dataset.py
```

---

## 🏋️ Training the Model

The training script `scripts/train_yolo26_segment.py` supports standard command-line arguments:

### CLI Arguments
| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `-m` | `yolo26n-seg.pt` | Pretrained YOLO model name or path |
| `--batch` | `-b` | `16` | Batch size |
| `--epochs` | `-e` | `100` | Number of training epochs |
| `--patience` | `-p` | `50` | Early stopping patience (epochs of no validation improvement) |
| `--name` | `-n` | `yolo26_id_card_seg` | Name of the W&B run and project output directory |
| `--test` | | *None* | Runs a quick test (1 epoch, 1 image, batch=1, workers=1) |

### Examples

#### 1. Quick Pipeline Test (Recommended before starting full train)
Run a quick dry-run with a single image and 1 epoch to ensure GPU, CUDA, and W&B logging are configured correctly:
```bash
python scripts/train_yolo26_segment.py --test
```

#### 2. Standard Training Run (Full Dataset)
Train a `YOLO26n-seg` model for 100 epochs, early-stopping patience of 20, batch size of 16, and run name `yolo26_cccd_run1`:
```bash
python scripts/train_yolo26_segment.py -m yolo26n-seg.pt -b 16 -e 100 -p 20 -n yolo26_cccd_run1
```

---

## 📊 Outputs & Monitoring

1. **W&B Live Dashboard**:
   - Metrics are uploaded in real-time. Only essential charts (losses, mAPs, precision, recall, learning rate, F1 score) are logged to minimize noise.
   - Standard console logs are captured and synced to the W&B **Logs** tab automatically.
2. **Local Output**:
   - Model weights and plots are saved locally under `model/segmentation/<run_name>/`.
   - Pretrained/Best weights are located at `model/segmentation/<run_name>/weights/best.pt`.
