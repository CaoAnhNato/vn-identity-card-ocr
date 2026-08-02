import os
import shutil
import sys
import yaml
from roboflow import Roboflow

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def parse_yaml_classes(yaml_path, target_names):
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found at {yaml_path}")
        return []
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    names = data.get("names", [])
    if isinstance(names, dict):
        # some formats have names as dict {0: 'class0', 1: 'class1'}
        target_ids = [int(k) for k, v in names.items() if v in target_names]
    else:
        # names is a list ['class0', 'class1']
        target_ids = [i for i, name in enumerate(names) if name in target_names]
        
    print(f"Dataset classes mapping: {names}")
    print(f"Target classes {target_names} mapped to IDs: {target_ids}")
    return target_ids

def filter_dataset(src_dir, dest_dir, target_ids):
    print(f"\nFiltering source dataset: {src_dir}")
    
    splits = ["train", "valid", "test"]
    stats = {split: 0 for split in splits}
    
    for split in splits:
        src_split_dir = os.path.join(src_dir, split)
        if not os.path.isdir(src_split_dir):
            continue
            
        src_images_dir = os.path.join(src_split_dir, "images")
        src_labels_dir = os.path.join(src_split_dir, "labels")
        
        if not os.path.isdir(src_images_dir) or not os.path.isdir(src_labels_dir):
            continue
            
        dest_images_dir = os.path.join(dest_dir, split, "images")
        dest_labels_dir = os.path.join(dest_dir, split, "labels")
        
        os.makedirs(dest_images_dir, exist_ok=True)
        os.makedirs(dest_labels_dir, exist_ok=True)
        
        for item in os.listdir(src_images_dir):
            if not os.path.isfile(os.path.join(src_images_dir, item)):
                continue
                
            base_name, _ = os.path.splitext(item)
            label_file_name = f"{base_name}.txt"
            src_label_path = os.path.join(src_labels_dir, label_file_name)
            
            if not os.path.isfile(src_label_path):
                continue
                
            # Check if label contains any target classes
            contains_target = False
            with open(src_label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if tokens:
                        try:
                            class_id = int(tokens[0])
                            if class_id in target_ids:
                                contains_target = True
                                break
                        except ValueError:
                            continue
            
            # If it contains at least one of the target classes, save it as a null image
            if contains_target:
                # Copy image
                src_img_path = os.path.join(src_images_dir, item)
                dest_img_path = os.path.join(dest_images_dir, item)
                shutil.copy2(src_img_path, dest_img_path)
                
                # Write an empty label file (completely empty to indicate null/no-target)
                dest_label_path = os.path.join(dest_labels_dir, label_file_name)
                with open(dest_label_path, "w", encoding="utf-8") as f:
                    pass
                
                stats[split] += 1
                
        print(f"  - Filtered {stats[split]} null images in '{split}' split")
        
    return stats

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    dest_dir = os.path.join(base_dir, "data", "null_dataset")
    
    # 1. Download dataset
    rf = Roboflow(api_key="S0EhROyNMg9SKh5qnTyA")
    print("\n--- Downloading Table Dataset ---")
    project = rf.workspace("celebalworkspace-bqx5k").project("table-03wsy")
    version = project.version(1)
    dataset = version.download("yolo26")
    src_path = dataset.location
    
    # 2. Get class IDs
    target_classes = ["book", "laptop", "mobile", "pen"]
    yaml_path = os.path.join(src_path, "data.yaml")
    target_ids = parse_yaml_classes(yaml_path, target_classes)
    
    if not target_ids:
        print("Error: Target classes not found in dataset mapping. Exiting...")
        return
        
    # 3. Filter and format
    print("\n--- Filtering and Formatting Null Dataset ---")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    stats = filter_dataset(src_path, dest_dir, target_ids)
    
    # 4. Generate statistics report
    total_images = sum(stats.values())
    print("\n--- Filtering Statistics ---")
    print(f"Train split null images: {stats.get('train', 0)}")
    print(f"Valid split null images: {stats.get('valid', 0)}")
    print(f"Test split null images: {stats.get('test', 0)}")
    print(f"Total null images filtered: {total_images}")
    print(f"Null dataset saved at: {dest_dir}")
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()
