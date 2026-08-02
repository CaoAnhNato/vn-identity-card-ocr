import os
import sys
import torch
import shutil
import argparse
from ultralytics import YOLO

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def estimate_vram(args):
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot measure GPU VRAM usage.")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"=== GPU DEVICE INFO ===")
    print(f"Device Name: {gpu_name}")
    print(f"Total VRAM: {total_memory:.2f} GB\n")
    
    # Base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Reset CUDA memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    initial_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    initial_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"Initial allocated memory: {initial_allocated:.2f} MB")
    print(f"Initial reserved memory:  {initial_reserved:.2f} MB\n")
    
    # 2. Load YOLO segmentation model
    model_name = args.model
    print(f"=== LOADING MODEL ===")
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    
    after_load_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    after_load_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    model_weights_mem = after_load_allocated - initial_allocated
    print(f"Memory after model load: {after_load_allocated:.2f} MB")
    print(f"Estimated model weights size in VRAM: {model_weights_mem:.2f} MB\n")
    
    # 3. Create a temporary dataset containing 1 image to perform a training step
    print("=== PREPARING TEMPORARY DATASET ===")
    temp_dataset_dir = os.path.join(base_dir, "data", "vram_test_temp")
    os.makedirs(os.path.join(temp_dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(temp_dataset_dir, "labels", "val"), exist_ok=True)
    
    merged_train_img_dir = os.path.join(base_dir, "data", "merged_roboflow_dataset", "train", "images")
    merged_train_lbl_dir = os.path.join(base_dir, "data", "merged_roboflow_dataset", "train", "labels")
    
    if not os.path.exists(merged_train_img_dir):
        print(f"ERROR: Merged dataset not found at {merged_train_img_dir}. Please run merge_datasets.py first.")
        sys.exit(1)
        
    train_imgs = [x for x in os.listdir(merged_train_img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    if not train_imgs:
        print("ERROR: No images found in merged_roboflow_dataset to run VRAM test.")
        sys.exit(1)
        
    # Copy first image and label
    image_name = train_imgs[0]
    base_name, _ = os.path.splitext(image_name)
    label_name = f"{base_name}.txt"
    
    shutil.copy2(os.path.join(merged_train_img_dir, image_name), os.path.join(temp_dataset_dir, "images", "train", image_name))
    shutil.copy2(os.path.join(merged_train_img_dir, image_name), os.path.join(temp_dataset_dir, "images", "val", image_name))
    shutil.copy2(os.path.join(merged_train_lbl_dir, label_name), os.path.join(temp_dataset_dir, "labels", "train", label_name))
    shutil.copy2(os.path.join(merged_train_lbl_dir, label_name), os.path.join(temp_dataset_dir, "labels", "val", label_name))
    
    # Write temp data.yaml
    data_yaml_content = f"""path: {temp_dataset_dir.replace(os.sep, '/')}
train: images/train
val: images/val
nc: 1
names: ['card']
"""
    data_yaml_path = os.path.join(temp_dataset_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"Created temporary configuration at: {data_yaml_path}\n")
    
    # 4. Disable W&B logging during measurement
    os.environ["WANDB_DISABLED"] = "true"
    
    # 5. Reset peak tracker and run 1 epoch, batch=1
    print("=== RUNNING TRAINING MEASUREMENT ===")
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        model.train(
            data=data_yaml_path,
            epochs=1,
            batch=1,
            imgsz=args.imgsz,
            device=0,
            workers=1,
            plots=False,
            save=False,
            val=False, # Disable validation to measure pure training memory
        )
        
        # 6. Retrieve Peak Statistics
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        
        print("\n" + "="*50)
        print("           VRAM ESTIMATION REPORT")
        print("="*50)
        print(f"Model File:          {model_name}")
        print(f"Image Size:          {args.imgsz}x{args.imgsz}")
        print(f"Base Model Size:     {model_weights_mem:.2f} MB")
        print(f"Peak VRAM Allocated: {peak_allocated:.2f} MB ({peak_allocated / 1024:.2f} GB)")
        print(f"Peak VRAM Reserved:  {peak_reserved:.2f} MB ({peak_reserved / 1024:.2f} GB)")
        print("-"*50)
        print(f"👉 Recommended minimum GPU VRAM for batch=1: {peak_reserved / 1024 + 0.5:.2f} GB")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nERROR: Failed to run training measurement: {e}\n")
    finally:
        # Cleanup temporary dataset gracefully to avoid Windows file lock PermissionError
        if os.path.exists(temp_dataset_dir):
            try:
                shutil.rmtree(temp_dataset_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dataset_dir} due to: {e}. You can delete it manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate VRAM consumption for YOLO segmentation model training.")
    parser.add_argument("-m", "--model", type=str, default="yolo26n-seg.pt", help="Path to model weights file.")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="Image resolution for training.")
    args = parser.parse_args()
    estimate_vram(args)
