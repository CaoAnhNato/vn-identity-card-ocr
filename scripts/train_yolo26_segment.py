import os
import sys
import shutil
import argparse
from dotenv import load_dotenv
load_dotenv() # Load variables from .env

import wandb
from ultralytics import YOLO
import albumentations as A

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Recommended on-the-fly Albumentations pipeline for card segmentation
custom_transforms = [
    A.SafeRotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.1),
    A.RandomShadow(p=0.2),
    # Occlusion transform (simulating fingers or obstacles covering parts of the card)
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.08, 0.25),
        hole_width_range=(0.05, 0.20),
        fill=0,
        p=0.3
    ),
]

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
        
        # Create a single-image dataset dynamically for testing from the merged dataset
        print("=== PREPARING SINGLE-IMAGE DATASET ===")
        single_dataset_dir = os.path.join(base_dir, "data", "merged_roboflow_dataset", "single_image_dataset")
        os.makedirs(os.path.join(single_dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "val"), exist_ok=True)

        merged_train_img_dir = os.path.join(base_dir, "data", "merged_roboflow_dataset", "train", "images")
        merged_train_lbl_dir = os.path.join(base_dir, "data", "merged_roboflow_dataset", "train", "labels")
        
        train_imgs = [x for x in os.listdir(merged_train_img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        if not train_imgs:
            raise FileNotFoundError("No images found in merged_roboflow_dataset/train/images to run test.")
            
        image_name = train_imgs[0]
        base_name, _ = os.path.splitext(image_name)
        label_name = f"{base_name}.txt"

        src_image = os.path.join(merged_train_img_dir, image_name)
        src_label = os.path.join(merged_train_lbl_dir, label_name)

        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "train", image_name))
        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "val", image_name))
        shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "train", label_name))
        shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "val", label_name))

        data_yaml_content = f"""path: {single_dataset_dir.replace(os.sep, '/')}
train: images/train
val: images/val

nc: 1
names: ['card']
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
        data_yaml_path = os.path.join(base_dir, "data", "merged_roboflow_dataset", "data.yaml")

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
        exist_ok=True,
        # On-the-fly Albumentations pipeline
        augmentations=custom_transforms,
        # Cache images in RAM for maximum training speed
        cache=True,
        # Disable native geometric/color augmentations since Albumentations handles them
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0
    )
    
    # 6. Upload model weights and evaluation curves/charts to WandB
    save_dir = os.path.join(base_dir, "model", "segmentation", run_name)
    if os.path.exists(save_dir):
        print("=== UPLOADING VAL/TEST ARTIFACTS TO WANDB ===")
        # 1. Upload best.pt
        best_weights_path = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_weights_path):
            print(f"Uploading best model weights: {best_weights_path}")
            wandb.save(best_weights_path, base_path=save_dir)
        else:
            print("Warning: best.pt weights file not found for upload.")
            
        # 2. Upload evaluation charts/curves (PNG files)
        for filename in os.listdir(save_dir):
            if filename.lower().endswith(".png"):
                filepath = os.path.join(save_dir, filename)
                print(f"Uploading evaluation chart: {filename}")
                curve_name = os.path.splitext(filename)[0]
                # Log as interactive image panel in WandB workspace
                wandb.log({f"curves/{curve_name}": wandb.Image(filepath)})
                # Save file directly under run files
                wandb.save(filepath, base_path=save_dir)
    else:
        print(f"Warning: save_dir not found at {save_dir}")
        
    # 7. Finish wandb run
    run.finish()
    print("=== TRAINING COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    args = parse_args()
    train(args)
