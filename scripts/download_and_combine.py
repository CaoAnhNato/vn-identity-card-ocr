import os
import shutil
import sys
from roboflow import Roboflow

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def process_source_dataset(src_dir, dest_dir, prefix):
    print(f"\nProcessing source dataset: {src_dir} (Prefix: {prefix})")
    
    # Check what split folders exist in the source
    possible_splits = {
        "train": "train",
        "valid": "val",
        "val": "val",
        "test": "val" # merge test split into val for validation richness
    }
    
    for src_split, dest_split in possible_splits.items():
        src_split_dir = os.path.join(src_dir, src_split)
        if not os.path.isdir(src_split_dir):
            continue
            
        src_images_dir = os.path.join(src_split_dir, "images")
        src_labels_dir = os.path.join(src_split_dir, "labels")
        
        if not os.path.isdir(src_images_dir):
            print(f"Warning: Images directory not found in {src_split_dir}")
            continue
            
        dest_images_dir = os.path.join(dest_dir, dest_split, "images")
        dest_labels_dir = os.path.join(dest_dir, dest_split, "labels")
        
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        
        copied_count = 0
        for item in os.listdir(src_images_dir):
            if not os.path.isfile(os.path.join(src_images_dir, item)):
                continue
                
            # Define new filename with prefix
            new_img_name = f"{prefix}{item}"
            src_img_path = os.path.join(src_images_dir, item)
            dest_img_path = os.path.join(dest_images_dir, new_img_name)
            
            # Copy image
            shutil.copy2(src_img_path, dest_img_path)
            
            # Look for corresponding label text file
            base_name, _ = os.path.splitext(item)
            label_file_name = f"{base_name}.txt"
            src_label_path = os.path.join(src_labels_dir, label_file_name)
            dest_label_path = os.path.join(dest_labels_dir, f"{prefix}{label_file_name}")
            
            if os.path.isfile(src_label_path):
                # Read, modify class to 0 (card), and write to destination
                with open(src_label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if tokens:
                        # Change class_id (first token) to '0'
                        tokens[0] = '0'
                        new_lines.append(" ".join(tokens) + "\n")
                
                with open(dest_label_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
            else:
                # If no label, it's considered background image (negative sample)
                # YOLO expects an empty txt file or no txt file for negative samples.
                # We write an empty file to be safe and explicit.
                with open(dest_label_path, "w", encoding="utf-8") as f:
                    pass
            
            copied_count += 1
            
        print(f"  - Copied {copied_count} files from '{src_split}' -> '{dest_split}'")

def main():
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    dest_dir = os.path.join(base_dir, "data", "combined_card_dataset")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Download Datasets
    rf = Roboflow(api_key="S0EhROyNMg9SKh5qnTyA")
    
    # Dataset 1: Business Cards
    print("\n--- Downloading Dataset 1 (Business Cards) ---")
    project1 = rf.workspace("sc-8qaqk").project("business-cards-n3hbf")
    version1 = project1.version(1)
    dataset1 = version1.download("yolo26")
    ds1_path = dataset1.location
    
    # Dataset 2: Temp Cards
    print("\n--- Downloading Dataset 2 (Temp Cards) ---")
    project2 = rf.workspace("temp-8tkum").project("temp-khmlk")
    version2 = project2.version(1)
    dataset2 = version2.download("yolo26")
    ds2_path = dataset2.location
    
    # ID-card-1 (Existing local dataset)
    ds_id_card_path = os.path.join(raw_dir, "ID-card-1")
    
    # 2. Combine and format
    print("\n--- Combining Datasets into data/combined_card_dataset ---")
    
    # Clean previous combined dataset if it exists to start fresh
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process ID-card-1
    if os.path.isdir(ds_id_card_path):
        process_source_dataset(ds_id_card_path, dest_dir, prefix="id1_")
    else:
        print(f"Warning: Existing ID-card-1 dataset not found at {ds_id_card_path}")
        
    # Process Dataset 1
    if os.path.isdir(ds1_path):
        process_source_dataset(ds1_path, dest_dir, prefix="bc_")
    else:
        print(f"Error: Dataset 1 not found at {ds1_path}")
        
    # Process Dataset 2
    if os.path.isdir(ds2_path):
        process_source_dataset(ds2_path, dest_dir, prefix="temp_")
    else:
        print(f"Error: Dataset 2 not found at {ds2_path}")
        
    # 3. Create data.yaml
    print("\n--- Generating data.yaml ---")
    yaml_path = os.path.join(dest_dir, "data.yaml")
    yaml_content = """# Combined Card Segmentation Dataset
train: train/images
val: val/images

nc: 1
names: ['card']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"data.yaml generated successfully at: {yaml_path}")
    print("\nAll datasets merged and formatted successfully!")

if __name__ == "__main__":
    main()
