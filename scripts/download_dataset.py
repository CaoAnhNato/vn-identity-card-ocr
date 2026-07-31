import os
import sys

# Ensure UTF-8 output encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from roboflow import Roboflow

def download():
    # Base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(dataset_dir, exist_ok=True)
    os.chdir(dataset_dir)

    print(f"Downloading dataset from Roboflow into: {dataset_dir}")
    rf = Roboflow(api_key="S0EhROyNMg9SKh5qnTyA")
    project = rf.workspace("loganqin").project("id-card-8apvj")
    version = project.version(1)
    dataset = version.download("yolo26")
    print("Download completed successfully!")
    print(f"Dataset location: {dataset.location}")

if __name__ == "__main__":
    download()
