import os
import sys
import shutil
import argparse
from dotenv import load_dotenv
load_dotenv() # Load variables from .env

import wandb
from ultralytics import YOLO

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26 instance segmentation model with custom W&B logging.")
    parser.add_argument("-m", "--model", type=str, default="yolo26n-seg.pt", help="Path or name of the YOLO model weights file.")
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("-n", "--name", type=str, default="yolo26_id_card_seg", help="Name of the training run / model saved to W&B.")
    parser.add_argument("-p", "--patience", type=int, default=50, help="Early stopping patience (epochs of no improvement before stopping).")
    parser.add_argument("--test", action="store_true", help="Run a quick training test with exactly 1 image and 1 epoch.")
    return parser.parse_args()

def train(args):
    # Base directory of project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we are running in test mode
    if args.test:
        print("=== RUNNING IN TEST MODE ===")
        # Override parameters for fast test run
        epochs = 1
        batch_size = 1
        workers = 1
        model_name = args.model
        run_name = f"{args.name}_test"
        
        # Create a single-image dataset dynamically for testing
        print("=== PREPARING SINGLE-IMAGE DATASET ===")
        single_dataset_dir = os.path.join(base_dir, "data", "raw", "ID-card-1", "single_image_dataset")
        os.makedirs(os.path.join(single_dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "val"), exist_ok=True)

        image_name = "01d3ee89-dfe6-49c4-a9ba-ab5f388a77f5-2022-06-03T04-42-31-409Z_jpg.rf.0d15a9215ee00e5602d4a417ad7b187e.jpg"
        label_name = "01d3ee89-dfe6-49c4-a9ba-ab5f388a77f5-2022-06-03T04-42-31-409Z_jpg.rf.0d15a9215ee00e5602d4a417ad7b187e.txt"

        src_image = os.path.join(base_dir, "data", "raw", "ID-card-1", "train", "images", image_name)
        src_label = os.path.join(base_dir, "data", "raw", "ID-card-1", "train", "labels", label_name)

        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "train", image_name))
        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "val", image_name))
        shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "train", label_name))
        shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "val", label_name))

        data_yaml_content = f"""path: {single_dataset_dir.replace(os.sep, '/')}
train: images/train
val: images/val

nc: 7
names: ['CCCD_BACK', 'CCCD_FRONT', 'CHIP_BACK', 'CHIP_FRONT', 'CMND_BACK', 'CMND_FRONT', 'PASSPORT']
"""
        data_yaml_path = os.path.join(single_dataset_dir, "data.yaml")
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write(data_yaml_content)
            
        print(f"Single image dataset created at: {single_dataset_dir}")
    else:
        print("=== RUNNING IN STANDARD TRAINING MODE ===")
        epochs = args.epochs
        batch_size = args.batch
        workers = 4
        model_name = args.model
        run_name = args.name
        data_yaml_path = os.path.join(base_dir, "data", "raw", "ID-card-1", "data.yaml")

    print(f"Data config path: {data_yaml_path}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Patience: {args.patience}")

    # 2. Login to wandb
    print("=== LOGGING IN TO WANDB ===")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")
        wandb.login(key=api_key)
    else:
        print("Warning: WANDB_API_KEY environment variable not found. Logging in via cached credentials or prompting...")

    # 3. Initialize wandb run with custom console wrapper to capture stdout/stderr training log
    print("=== INITIALIZING WANDB RUN ===")
    run = wandb.init(
        entity="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
        project="ID_Card_VN",
        name=run_name,
        config={
            "learning_rate": 0.01,
            "architecture": model_name,
            "dataset": "ID_Card_VN" if not args.test else "ID_Card_VN_Single_Image",
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": args.patience,
        },
        settings=wandb.Settings(console="wrap")
    )

    print("=== START TRAINING YOLO SEGMENT MODEL ===")
    
    # Load YOLO segmentation model
    model = YOLO(model_name)

    # 4. Remove default ultralytics wandb callbacks to prevent logging junk files/plots
    for event, callbacks in model.callbacks.items():
        model.callbacks[event] = [
            cb for cb in callbacks
            if "wandb" not in cb.__module__ and "wandb" not in cb.__name__
        ]

    # 5. Register custom callback to log only essential metrics to wandb
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        
        # Extract loss metrics
        tloss = trainer.tloss
        loss_items = trainer.label_loss_items(tloss, prefix="train")
        
        # Extract learning rates
        lr_dict = trainer.lr
        
        # Extract validation metrics
        val_metrics = trainer.metrics
        
        # Construct essential log dictionary
        log_data = {"epoch": epoch}
        
        # Add training losses
        for k, v in loss_items.items():
            log_data[k] = v
            
        # Add validation losses (if any)
        for k, v in val_metrics.items():
            if "loss" in k:
                log_data[k] = v
                
        # Add learning rates
        for k, v in lr_dict.items():
            log_data[k] = v
            
        # Add mAP, Precision, Recall
        for k, v in val_metrics.items():
            if any(metric in k for metric in ["precision", "recall", "mAP"]):
                log_data[k] = v
                
        # Compute and log F1 score (Box and Mask/Seg)
        pb = val_metrics.get("metrics/precision(B)")
        rb = val_metrics.get("metrics/recall(B)")
        if pb is not None and rb is not None:
            log_data["metrics/f1(B)"] = 2 * (pb * rb) / (pb + rb + 1e-8)
            
        pm = val_metrics.get("metrics/precision(M)")
        rm = val_metrics.get("metrics/recall(M)")
        if pm is not None and rm is not None:
            log_data["metrics/f1(M)"] = 2 * (pm * rm) / (pm + rm + 1e-8)
            
        # Log to wandb
        wandb.log(log_data)
        print(f"\n[Wandb Logged] Epoch {epoch} Metrics: {log_data}\n")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Train model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=0,  # GPU 0
        workers=workers,
        patience=args.patience,
        project=os.path.join(base_dir, "model", "segmentation"),
        name=run_name,
        exist_ok=True
    )
    
    # 6. Finish wandb run
    run.finish()
    print("=== TRAINING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    args = parse_args()
    train(args)
