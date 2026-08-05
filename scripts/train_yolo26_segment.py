import os
import sys
import shutil
import argparse
from dotenv import load_dotenv
load_dotenv() # Load variables from .env

import wandb
from ultralytics import YOLO
import albumentations as A
import random

# Monkey-patch Albumentations to pass instance segmentation masks to custom transforms
from ultralytics.data.augment import Albumentations
import cv2

original_albumentations_call = Albumentations.__call__

def patched_albumentations_call(self, labels):
    if self.transform is None or random.random() > self.p:
        return labels

    im = labels["img"]
    if im.shape[2] != 3:  # Only apply Albumentation on 3-channel images
        return labels

    # Reconstruct binary mask from instances.segments (YOLO instance segmentation)
    h, w = im.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    instances = labels.get("instances")
    has_mask = False
    if instances is not None and hasattr(instances, "segments") and len(instances.segments) > 0:
        for seg in instances.segments:
            if np.all(seg == 0):  # Skip empty/padded segments
                continue
            pts = seg.copy()
            if instances.normalized:
                pts[:, 0] *= w
                pts[:, 1] *= h
            pts = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            has_mask = True

    if has_mask:
        new = self.transform(image=im, mask=mask)
        labels["img"] = new["image"]
    else:
        labels["img"] = self.transform(image=im)["image"]

    return labels

Albumentations.__call__ = patched_albumentations_call

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Custom Albumentations transform to simulate finger occlusion on the card's edge/boundary
# This prevents IndexError mismatches in YOLO's dataloader by leaving bboxes/masks unmodified,
# while drawing black boxes centered on the card boundary to teach the model to segment the entire card.
import numpy as np

class CardEdgeOcclusion(A.DualTransform):
    def __init__(self, num_holes_range=(1, 3), hole_height_range=(0.08, 0.25), hole_width_range=(0.05, 0.20), fill=0, p=0.5):
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill

    def get_params_dependent_on_data(self, params, data):
        image = data["image"]
        h, w = image.shape[:2]
        num_holes = self.py_random.randint(self.num_holes_range[0], self.num_holes_range[1])
        holes = []
        mask = data.get("mask")
        contour_points = []
        if mask is not None:
            import cv2
            if mask.dtype != np.uint8:
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
            else:
                mask_bin = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                for pt in c:
                    contour_points.append(pt[0])
        if contour_points:
            for _ in range(num_holes):
                idx = self.py_random.randint(0, len(contour_points) - 1)
                cx, cy = contour_points[idx]
                min_h = int(self.hole_height_range[0] * h)
                max_h = int(self.hole_height_range[1] * h)
                min_w = int(self.hole_width_range[0] * w)
                max_w = int(self.hole_width_range[1] * w)
                hole_h = self.py_random.randint(max(1, min_h), max(2, max_h))
                hole_w = self.py_random.randint(max(1, min_w), max(2, max_w))
                ymin = max(0, cy - hole_h // 2)
                ymax = min(h, cy + hole_h // 2)
                xmin = max(0, cx - hole_w // 2)
                xmax = min(w, cx + hole_w // 2)
                holes.append((ymin, ymax, xmin, xmax))
        else:
            for _ in range(num_holes):
                min_h = int(self.hole_height_range[0] * h)
                max_h = int(self.hole_height_range[1] * h)
                min_w = int(self.hole_width_range[0] * w)
                max_w = int(self.hole_width_range[1] * w)
                hole_h = self.py_random.randint(max(1, min_h), max(2, max_h))
                hole_w = self.py_random.randint(max(1, min_w), max(2, max_w))
                cy = self.py_random.randint(hole_h // 2, h - hole_h // 2)
                cx = self.py_random.randint(hole_w // 2, w - hole_w // 2)
                ymin = max(0, cy - hole_h // 2)
                ymax = min(h, cy + hole_h // 2)
                xmin = max(0, cx - hole_w // 2)
                xmax = min(w, cx + hole_w // 2)
                holes.append((ymin, ymax, xmin, xmax))
        return {"holes": holes}

    def apply(self, img, holes=None, **params):
        if holes is None:
            return img
        img_out = img.copy()
        for ymin, ymax, xmin, xmax in holes:
            img_out[ymin:ymax, xmin:xmax] = self.fill
        return img_out

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_to_masks(self, masks, **params):
        return masks

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes

    def get_transform_init_args_names(self):
        return ("num_holes_range", "hole_height_range", "hole_width_range", "fill")

# Recommended on-the-fly Albumentations pipeline for card segmentation (Non-spatial transforms only)
custom_transforms = [
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.1),
    A.RandomShadow(p=0.2),
    # Edge occlusion transform simulating fingers holding the card edges
    CardEdgeOcclusion(
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
    parser.add_argument("-d", "--data", type=str, default=None, help="Path to data.yaml file.")
    parser.add_argument("--project", type=str, default="ID_Card_VN", help="W&B project name.")
    parser.add_argument("-p", "--patience", type=int, default=50, help="Early stopping patience (epochs of no improvement before stopping).")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler during training.")
    parser.add_argument("--freeze", type=int, default=None, help="Number of initial layers to freeze (e.g. 10 to freeze backbone).")
    parser.add_argument("--staged", action="store_true", help="Enable staged training: freeze backbone first, then unfreeze and fine-tune.")
    parser.add_argument("--freeze_epochs", type=int, default=30, help="Number of epochs to train with frozen backbone in staged mode.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW", "SGD", "RMSProp", "auto"], help="Optimizer to use for training. Defaults to 'AdamW' to prevent lr0 override.")
    parser.add_argument("--mosaic", type=float, default=0.2, help="Mosaic augmentation probability. Defaults to 0.2.")
    parser.add_argument("--test", action="store_true", help="Run a quick training test with exactly 1 image and 1 epoch.")
    return parser.parse_args()

def train(args):
    # Base directory of project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine the data.yaml path first
    if args.data:
        data_yaml_path = os.path.abspath(args.data)
    else:
        data_yaml_path = os.path.join(base_dir, "data", "merged_roboflow_dataset", "data.yaml")

    # Check if we are running in test mode
    if args.test:
        print("=== RUNNING IN TEST MODE ===")
        os.environ["WANDB_MODE"] = "offline"
        # Override parameters for fast test run
        epochs = 1
        batch_size = 1
        workers = 1
        model_name = args.model
        run_name = f"{args.name}_test"
        
        # Create a single-image dataset dynamically for testing
        dataset_dir = os.path.dirname(data_yaml_path)
        print("=== PREPARING SINGLE-IMAGE DATASET ===")
        single_dataset_dir = os.path.join(dataset_dir, "single_image_dataset")
        os.makedirs(os.path.join(single_dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(single_dataset_dir, "labels", "val"), exist_ok=True)

        merged_train_img_dir = os.path.join(dataset_dir, "train", "images")
        merged_train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
        
        train_imgs = [x for x in os.listdir(merged_train_img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        if not train_imgs:
            raise FileNotFoundError(f"No images found in {merged_train_img_dir} to run test.")
            
        image_name = train_imgs[0]
        base_name, _ = os.path.splitext(image_name)
        label_name = f"{base_name}.txt"

        src_image = os.path.join(merged_train_img_dir, image_name)
        src_label = os.path.join(merged_train_lbl_dir, label_name)

        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "train", image_name))
        shutil.copy2(src_image, os.path.join(single_dataset_dir, "images", "val", image_name))
        if os.path.exists(src_label):
            shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "train", label_name))
            shutil.copy2(src_label, os.path.join(single_dataset_dir, "labels", "val", label_name))
        else:
            with open(os.path.join(single_dataset_dir, "labels", "train", label_name), "w") as lf:
                pass
            with open(os.path.join(single_dataset_dir, "labels", "val", label_name), "w") as lf:
                pass

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
        project=args.project,
        name=run_name,
        config={
            "learning_rate": 0.01,
            "architecture": model_name,
            "dataset": args.project if not args.test else f"{args.project}_Single_Image",
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": args.patience,
        },
        settings=wandb.Settings(console="wrap")
    )

    print("=== START TRAINING YOLO SEGMENT MODEL ===")
    
    # 4. Define local custom callback function with epoch offset support
    epoch_offset = [0] # List wrapper to mutate in outer scope

    def on_fit_epoch_end(trainer):
        epoch = epoch_offset[0] + trainer.epoch + 1
        
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

    # Helper function to remove default W&B callbacks and add our custom one
    def configure_callbacks(m):
        for event, callbacks in m.callbacks.items():
            m.callbacks[event] = [
                cb for cb in callbacks
                if "wandb" not in cb.__module__ and "wandb" not in cb.__name__
            ]
        m.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # 5. Execute Staged or Standard Training
    if args.staged:
        print("=== STAGED TRAINING ENABLED ===")
        if args.test:
            stage1_epochs = 1
            stage2_epochs = 1
        else:
            stage1_epochs = args.freeze_epochs
            stage2_epochs = max(1, epochs - args.freeze_epochs)
            
        print(f"Stage 1 (Freeze Backbone): {stage1_epochs} epochs")
        print(f"Stage 2 (Fine-tuning):      {stage2_epochs} epochs")
        
        # --- STAGE 1: Freeze training ---
        print("\n--- STARTING STAGE 1: Freeze Backbone ---")
        model = YOLO(model_name)
        configure_callbacks(model)
        
        freeze_layers = args.freeze if args.freeze is not None else 10
        # Determine Stage 1 learning rate: 0.002 is optimal for AdamW/Adam in YOLO, 0.01 is for SGD.
        stage1_lr = 0.002 if args.optimizer in ["Adam", "AdamW"] else 0.01
        model.train(
            data=data_yaml_path,
            epochs=stage1_epochs,
            imgsz=640,
            batch=batch_size,
            device=0,
            workers=workers,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=stage1_lr,
            cos_lr=args.cos_lr,
            freeze=freeze_layers,
            project=os.path.join(base_dir, "model", "segmentation"),
            name=run_name,
            exist_ok=True,
            augmentations=custom_transforms,
            cache=True,
            mosaic=args.mosaic,
            degrees=30.0, translate=0.06, scale=0.1, shear=0.0, perspective=0.05,
            flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0
        )
        
        # --- STAGE 2: Fine-tune training ---
        print("\n--- STARTING STAGE 2: Fine-tuning ---")
        best_weights_stage1 = os.path.join(base_dir, "model", "segmentation", run_name, "weights", "best.pt")
        if not os.path.exists(best_weights_stage1):
            print(f"Warning: Stage 1 weights not found at {best_weights_stage1}, using model name instead.")
            best_weights_stage1 = model_name
            
        model = YOLO(best_weights_stage1)
        configure_callbacks(model)
        
        epoch_offset[0] = stage1_epochs
        model.train(
            data=data_yaml_path,
            epochs=stage2_epochs,
            imgsz=640,
            batch=batch_size,
            device=0,
            workers=workers,
            patience=args.patience,
            optimizer=args.optimizer,
            cos_lr=args.cos_lr,
            freeze=None, # Unfreeze all layers
            lr0=0.0002,  # Lower learning rate for fine-tuning
            warmup_epochs=0.0, # Disable warmup in Stage 2 to prevent bias learning rate spikes
            project=os.path.join(base_dir, "model", "segmentation"),
            name=run_name,
            exist_ok=True,
            augmentations=custom_transforms,
            cache=True,
            mosaic=args.mosaic,
            degrees=30.0, translate=0.06, scale=0.1, shear=0.0, perspective=0.05,
            flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0
        )
    else:
        print("=== STANDARD TRAINING ENABLED ===")
        model = YOLO(model_name)
        configure_callbacks(model)
        
        # Determine learning rate: 0.002 is optimal for AdamW/Adam in YOLO, 0.01 is for SGD.
        std_lr = 0.002 if args.optimizer in ["Adam", "AdamW"] else 0.01
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=0,
            workers=workers,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=std_lr,
            cos_lr=args.cos_lr,
            freeze=args.freeze,
            project=os.path.join(base_dir, "model", "segmentation"),
            name=run_name,
            exist_ok=True,
            augmentations=custom_transforms,
            cache=True,
            mosaic=args.mosaic,
            degrees=30.0, translate=0.06, scale=0.1, shear=0.0, perspective=0.05,
            flipud=0.0, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0
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
