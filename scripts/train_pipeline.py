import os
import sys
import shutil
import argparse
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Load env variables
load_dotenv(os.path.join(base_dir, ".env"))

def parse_pipeline_args():
    parser = argparse.ArgumentParser(description="End-to-End 2-Phase YOLO26 Training Pipeline.")
    parser.add_argument("-m", "--model", type=str, default="yolo26n-seg.pt", help="Path or name of the starting pre-trained weights.")
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs_phase1", type=int, default=50, help="Epochs for Phase 1 (Business Card).")
    parser.add_argument("--epochs_phase2", type=int, default=100, help="Epochs for Phase 2 (ID Card).")
    parser.add_argument("-p", "--patience", type=int, default=30, help="Patience for early stopping.")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW", "SGD", "RMSProp", "auto"], help="Optimizer.")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading datasets if folders exist.")
    parser.add_argument("--test", action="store_true", help="Run a quick 1-epoch pipeline test on 1 image.")
    return parser.parse_args()

def download_roboflow_dataset(api_key, workspace, project_name, version_num, target_dir_name, skip_download=False):
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    target_path = os.path.join(data_dir, target_dir_name)
    
    if skip_download and os.path.exists(target_path):
        print(f"Dataset already exists at {target_path}. Skipping download as requested.")
        return target_path

    # Check if directory exists
    if os.path.exists(target_path):
        print(f"Dataset directory already exists at {target_path}. Reusing existing files.")
        return target_path

    print(f"\n=== Downloading dataset '{project_name}' (v{version_num}) from Roboflow ===")
    
    # Temporarily change directory to data_dir for Roboflow download behavior
    old_cwd = os.getcwd()
    os.chdir(data_dir)
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
        dataset = version.download("yolo26")
        
        src_path = os.path.abspath(dataset.location)
        if src_path != os.path.abspath(target_path):
            print(f"Mapping downloaded dataset to: {target_path}")
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            os.rename(src_path, target_path)
    except Exception as e:
        print(f"Error downloading dataset {project_name}: {e}")
        sys.exit(1)
    finally:
        os.chdir(old_cwd)
        
    return target_path

def fix_data_yaml(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"Warning: data.yaml not found at {yaml_path}")
        return
        
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error parsing {yaml_path}: {e}")
            return
            
    # Fix paths for YOLO compatibility
    data["path"] = os.path.abspath(dataset_dir).replace(os.sep, "/")
    for key in ["train", "val", "test"]:
        if key in data:
            val = data[key]
            if val.startswith("../"):
                data[key] = val[3:]
                
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"Configured data.yaml path: {yaml_path}")

def main():
    args = parse_pipeline_args()
    
    # 1. API Keys
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY", "S0EhROyNMg9SKh5qnTyA").strip().strip('"').strip("'")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY is not set in environment or .env file.")
        
    # Write ROBOFLOW_API_KEY to .env if not present
    env_file_path = os.path.join(base_dir, ".env")
    if not os.path.exists(env_file_path):
        with open(env_file_path, "w") as f:
            f.write(f"ROBOFLOW_API_KEY={roboflow_api_key}\n")
    else:
        with open(env_file_path, "r") as f:
            env_content = f.read()
        if "ROBOFLOW_API_KEY" not in env_content:
            with open(env_file_path, "a") as f:
                f.write(f"\nROBOFLOW_API_KEY={roboflow_api_key}\n")

    # 2. Download datasets
    print("=== Phase 1 & 2 Datasets Check/Download ===")
    bc_dir = download_roboflow_dataset(
        api_key=roboflow_api_key,
        workspace="detecobject",
        project_name="business_card-gu85b",
        version_num=1,
        target_dir_name="business_cards_dataset",
        skip_download=args.skip_download
    )
    fix_data_yaml(bc_dir)
    
    id_dir = download_roboflow_dataset(
        api_key=roboflow_api_key,
        workspace="detecobject",
        project_name="id_card-rmuep",
        version_num=1,
        target_dir_name="id_cards_dataset",
        skip_download=args.skip_download
    )
    fix_data_yaml(id_dir)

    # Import train logic
    from scripts.train_yolo26_segment import train as train_func
    
    # 3. Phase 1: Business Card Pre-training
    print("\n=========================================")
    print("=== PHASE 1: BUSINESS CARD PRE-TRAINING ===")
    print("=========================================")
    
    p1_model_name = args.model
    p1_epochs = 1 if args.test else args.epochs_phase1
    p1_batch = 1 if args.test else args.batch
    p1_run_name = "yolo26_business_card_pretrain"
    
    p1_args = argparse.Namespace(
        model=p1_model_name,
        batch=p1_batch,
        epochs=p1_epochs,
        name=p1_run_name,
        data=os.path.join(bc_dir, "data.yaml"),
        project="Business_Card_Pretrain",
        patience=args.patience,
        cos_lr=args.cos_lr,
        freeze=None,
        staged=False,
        freeze_epochs=30,
        optimizer=args.optimizer,
        mosaic=0.0,
        test=args.test
    )
    
    print(f"Starting Phase 1 with base model: {p1_model_name}")
    train_func(p1_args)
    
    # Determine the best weights path from Phase 1
    p1_best_weights = os.path.join(base_dir, "model", "segmentation", p1_run_name if not args.test else f"{p1_run_name}_test", "weights", "best.pt")
    if not os.path.exists(p1_best_weights):
        print(f"Warning: Best weights from Phase 1 not found at {p1_best_weights}. Using starting weights '{args.model}' for Phase 2.")
        p1_best_weights = args.model
    else:
        print(f"Successfully located best weights from Phase 1: {p1_best_weights}")
        
    # 4. Phase 2: ID Card Fine-tuning
    print("\n=========================================")
    print("=== PHASE 2: ID CARD FINE-TUNING ========")
    print("=========================================")
    
    p2_epochs = 1 if args.test else args.epochs_phase2
    p2_batch = 1 if args.test else args.batch
    p2_run_name = "yolo26_id_card_finetune"
    
    p2_args = argparse.Namespace(
        model=p1_best_weights,
        batch=p2_batch,
        epochs=p2_epochs,
        name=p2_run_name,
        data=os.path.join(id_dir, "data.yaml"),
        project="ID_Card_VN",
        patience=args.patience,
        cos_lr=args.cos_lr,
        freeze=None,
        staged=False,
        freeze_epochs=30,
        optimizer=args.optimizer,
        mosaic=0.0,
        test=args.test
    )
    
    print(f"Starting Phase 2 with Phase 1 weights: {p1_best_weights}")
    train_func(p2_args)
    
    print("\n=========================================")
    print("=== PIPELINE RUN COMPLETED SUCCESSFULLY ===")
    print("=========================================")

if __name__ == "__main__":
    main()
