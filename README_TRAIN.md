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

The combined card and negative background dataset is hosted on Roboflow. Run the download script to automatically download and map it into `data/merged_roboflow_dataset`:

```bash
python scripts/download_dataset.py
```

*(Note: Empty background/negative images are included with corresponding empty `.txt` label files to teach the model to ignore background clutter and eliminate False Positives).*

### Step 4: Estimate GPU VRAM & Optimize Batch Size (Optional)

Before starting full training, you can run the VRAM estimation script to determine the peak GPU memory usage and choose the optimal batch size for your GPU container:

```bash
python scripts/estimate_vram.py --model yolo26n-seg.pt --imgsz 640
```

This script will output the base model size, peak allocated memory, and recommended batch sizes.

---

## 🏋️ Training the Model

The training script `scripts/train_yolo26_segment.py` supports standard command-line arguments:

### CLI Arguments

| Argument         | Short  | Default              | Description                                                                |
| :--------------- | :----- | :------------------- | :------------------------------------------------------------------------- |
| `--model`        | `-m`   | `yolo26n-seg.pt`     | Pretrained YOLO model name or path                                         |
| `--batch`        | `-b`   | `16`                 | Batch size                                                                 |
| `--epochs`       | `-e`   | `100`                | Number of training epochs                                                  |
| `--patience`     | `-p`   | `50`                 | Early stopping patience (epochs of no validation improvement)              |
| `--cos_lr`       |        | *None*               | Uses cosine learning rate scheduler during training                        |
| `--freeze`       |        | *None*               | Number of layers to freeze from the beginning (e.g. 10 to freeze backbone) |
| `--staged`       |        | *None*               | Enables staged training (freeze backbone first, then unfreeze & fine-tune) |
| `--freeze_epochs`|        | `30`                 | Number of epochs to train with frozen backbone in staged mode              |
| `--name`         | `-n`   | `yolo26_id_card_seg` | Name of the W&B run and project output directory                           |
| `--test`         |        | *None*               | Runs a quick test (1 epoch, 1 image, batch=1, workers=1)                   |

### Examples

#### 1. Quick Pipeline Test (Recommended before starting full train)

Run a quick dry-run with a single image and 1 epoch to ensure GPU, CUDA, and W&B logging are configured correctly:

```bash
python scripts/train_yolo26_segment.py --test
```

#### 2. Standard Training Run (Backbone Frozen Permanently)

Train a `YOLO26n-seg` model for 100 epochs, early-stopping patience of 20, batch size of 16, freeze the backbone (first 10 layers) permanently, and use a cosine learning rate scheduler:

```bash
python scripts/train_yolo26_segment.py -m yolo26n-seg.pt -b 16 -e 100 -p 20 --cos_lr --freeze 10 -n yolo26_cccd_standard
```

#### 3. Staged Training Run (Recommended - Warm-up then Fine-tune)

Train a `YOLO26n-seg` model for 100 epochs, early-stopping patience of 20, batch size of 16, using staged training (Stage 1: freeze backbone for 30 epochs; Stage 2: unfreeze and fine-tune for 70 epochs), and a cosine learning rate scheduler:

```bash
python scripts/train_yolo26_segment.py -m yolo26n-seg.pt -b 16 -e 100 -p 20 --staged --freeze_epochs 30 --cos_lr -n yolo26_cccd_staged
```

*(Note: Images are cached in RAM (`cache=True`) for maximum data loading speed).*

---

## 📊 Outputs & Monitoring

1. **W&B Live Dashboard**:
   - Metrics are uploaded in real-time. Only essential charts (losses, mAPs, precision, recall, learning rate, F1 score) are logged to minimize noise.
   - Standard console logs are captured and synced to the W&B **Logs** tab automatically.
   - At the end of training, the model's best weights (`best.pt`) and evaluation charts (e.g. `BoxF1_curve.png`, `MaskF1_curve.png`, etc.) are automatically uploaded directly to the W&B Run Files and Media workspace.
2. **Local Output**:
   - Model weights and plots are saved locally under `model/segmentation/<run_name>/`.
   - Pretrained/Best weights are located at `model/segmentation/<run_name>/weights/best.pt`.
