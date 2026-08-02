import os
import sys
import shutil

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def merge_split(src_img_dir, src_label_dir, dest_img_dir, dest_label_dir):
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    if not os.path.exists(src_img_dir):
        print(f"Warning: Source images directory {src_img_dir} does not exist.")
        return 0
        
    copied = 0
    img_files = [x for x in os.listdir(src_img_dir) if x.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    for img_name in img_files:
        src_img_path = os.path.join(src_img_dir, img_name)
        dest_img_path = os.path.join(dest_img_dir, img_name)
        
        # Copy image
        shutil.copy2(src_img_path, dest_img_path)
        
        # Copy corresponding label if it exists, otherwise write empty file
        base_name, _ = os.path.splitext(img_name)
        label_name = f"{base_name}.txt"
        src_label_path = os.path.join(src_label_dir, label_name)
        dest_label_path = os.path.join(dest_label_dir, label_name)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dest_label_path)
        else:
            # Create empty label file (Roboflow format for null backgrounds)
            with open(dest_label_path, "w", encoding="utf-8") as f:
                pass
                
        copied += 1
    return copied

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    card_dataset_dir = os.path.join(data_dir, "combined_card_dataset")
    null_dataset_dir = os.path.join(data_dir, "null_dataset")
    dest_dataset_dir = os.path.join(data_dir, "merged_roboflow_dataset")
    
    print("=== STARTING DATASET MERGE ===")
    print(f"Card Dataset: {card_dataset_dir}")
    print(f"Null Dataset: {null_dataset_dir}")
    print(f"Target Merged Dataset: {dest_dataset_dir}")
    
    # 1. Clean existing merged dataset directory if it exists to start fresh
    if os.path.exists(dest_dataset_dir):
        print(f"Removing existing directory: {dest_dataset_dir}")
        shutil.rmtree(dest_dataset_dir)
        
    os.makedirs(dest_dataset_dir, exist_ok=True)
    
    # 2. Merge Train: Card train + Null train
    print("\n--- Merging Train Split ---")
    c_train = merge_split(
        os.path.join(card_dataset_dir, "train", "images"),
        os.path.join(card_dataset_dir, "train", "labels"),
        os.path.join(dest_dataset_dir, "train", "images"),
        os.path.join(dest_dataset_dir, "train", "labels")
    )
    print(f"Copied {c_train} images from Card train.")
    
    n_train = merge_split(
        os.path.join(null_dataset_dir, "train", "images"),
        os.path.join(null_dataset_dir, "train", "labels"),
        os.path.join(dest_dataset_dir, "train", "images"),
        os.path.join(dest_dataset_dir, "train", "labels")
    )
    print(f"Copied {n_train} images from Null train.")
    print(f"Total train images: {c_train + n_train}")
    
    # 3. Merge Valid: Card val + Null valid (Roboflow expects folder name 'valid')
    print("\n--- Merging Valid Split ---")
    c_val = merge_split(
        os.path.join(card_dataset_dir, "val", "images"),
        os.path.join(card_dataset_dir, "val", "labels"),
        os.path.join(dest_dataset_dir, "valid", "images"),
        os.path.join(dest_dataset_dir, "valid", "labels")
    )
    print(f"Copied {c_val} images from Card val.")
    
    n_val = merge_split(
        os.path.join(null_dataset_dir, "valid", "images"),
        os.path.join(null_dataset_dir, "valid", "labels"),
        os.path.join(dest_dataset_dir, "valid", "images"),
        os.path.join(dest_dataset_dir, "valid", "labels")
    )
    print(f"Copied {n_val} images from Null valid.")
    print(f"Total valid images: {c_val + n_val}")
    
    # 4. Merge Test: Null test (Card dataset doesn't have test split)
    print("\n--- Merging Test Split ---")
    n_test = merge_split(
        os.path.join(null_dataset_dir, "test", "images"),
        os.path.join(null_dataset_dir, "test", "labels"),
        os.path.join(dest_dataset_dir, "test", "images"),
        os.path.join(dest_dataset_dir, "test", "labels")
    )
    print(f"Copied {n_test} images from Null test.")
    print(f"Total test images: {n_test}")
    
    # 5. Create data.yaml
    data_yaml_content = """# Merged Card & Null Dataset for Instance Segmentation
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['card']
"""
    data_yaml_path = os.path.join(dest_dataset_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    print(f"\nCreated data.yaml at: {data_yaml_path}")
    print("=== DATASET MERGE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
