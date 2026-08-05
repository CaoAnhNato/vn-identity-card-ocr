import os
import sys
import shutil
from roboflow import Roboflow

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def download():
    # Base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    print(f"Downloading dataset from Roboflow into: {data_dir}")
    rf = Roboflow(api_key="S0EhROyNMg9SKh5qnTyA")
    project = rf.workspace("detecobject").project("id_card_segment-dv2fc")
    version = project.version(2)
    dataset = version.download("yolo26")
    print("Download completed successfully!")
    print(f"Dataset location: {dataset.location}")

    # Standard target path for the training scripts
    target_path = os.path.join(data_dir, "merged_roboflow_dataset")
    if os.path.abspath(dataset.location) != os.path.abspath(target_path):
        if os.path.exists(target_path):
            print(f"Cleaning up existing target directory: {target_path}")
            try:
                if os.path.islink(target_path):
                    os.unlink(target_path)
                elif os.path.isdir(target_path):
                    import stat
                    def handle_remove_readonly(func, path, exc_info):
                        # Clear read-only attribute and retry
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    shutil.rmtree(target_path, onerror=handle_remove_readonly)
                else:
                    os.remove(target_path)
            except Exception as e:
                print(f"Warning: Could not remove {target_path} automatically due to: {e}. You may need to delete it manually.")

        print(f"Mapping downloaded dataset directory to standard path: {target_path}")
        try:
            # Try symlink first (needs admin on Windows, fallback if fails)
            if sys.platform == "win32":
                import subprocess
                subprocess.run(['mklink', '/D', target_path, dataset.location], shell=True, check=True)
            else:
                os.symlink(dataset.location, target_path)
            print("Symlink created successfully.")
        except Exception as e:
            print(f"Symlink failed ({e}), renaming directory instead...")
            # Fallback to renaming/moving
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            os.rename(dataset.location, target_path)
            print("Directory renamed successfully.")

if __name__ == "__main__":
    download()
