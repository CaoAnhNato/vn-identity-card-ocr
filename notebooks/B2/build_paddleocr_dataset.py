"""
Input: 1 folder chứa ảnh + json labelme (points, text) - đầu ra từ script OCR trước đó.
Output: dataset format PaddleOCR (detection + recognition), dict sinh từ text thật trong data.

    out_dir/
        det/
            images/
            train_det.txt
            val_det.txt
        rec/
            images/
            rec_gt_train.txt
            rec_gt_val.txt
        dict/
            word_dict.txt

Usage:
    python build_paddleocr_dataset.py --folder /path/to/images_with_json --out /path/to/dataset --val-ratio 0.1
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_quad(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    rect = order_points(poly.astype("float32"))
    (tl, tr, br, bl) = rect
    w = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)), 1)
    h = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)), 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def load_pairs(folder: Path):
    pairs = []
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Bỏ qua {img_path.name}: không có json tương ứng")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = [s for s in data.get("shapes", []) if s.get("text", "").strip()]
        if not shapes:
            print(f"Bỏ qua {img_path.name}: không có shape nào có text")
            continue
        pairs.append((img_path, shapes))
    return pairs


def build_dataset(folder: str, out_dir: str, val_ratio: float, seed: int):
    folder_path = Path(folder)
    out_path = Path(out_dir)

    det_img_dir = out_path / "det" / "images"
    rec_img_dir = out_path / "rec" / "images"
    dict_dir = out_path / "dict"
    for d in (det_img_dir, rec_img_dir, dict_dir):
        d.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(folder_path)
    if not pairs:
        print("Không có cặp ảnh/json hợp lệ nào.")
        return

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    charset = set()
    det_lines = {"train": [], "val": []}
    rec_lines = {"train": [], "val": []}

    def process_split(split_pairs, split_name):
        for img_path, shapes in split_pairs:
            dst_img = det_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            det_annots = []
            image_bgr = None

            for i, shape in enumerate(shapes):
                text = shape["text"]
                if "\t" in text or "\n" in text:
                    print(f"CẢNH BÁO: text chứa tab/xuống dòng ở {img_path.name} "
                          f"(shape {i}): {text!r} -> đã thay bằng khoảng trắng")
                    text = text.replace("\t", " ").replace("\n", " ")
                points = shape["points"]
                charset.update(text)

                int_points = [[int(round(x)), int(round(y))] for x, y in points]
                det_annots.append({"transcription": text, "points": int_points})

                if image_bgr is None:
                    image_bgr = cv2.imread(str(img_path))
                poly = np.array(points, dtype="float32")
                crop = crop_quad(image_bgr, poly)
                if crop.size == 0:
                    continue
                crop_name = f"{img_path.stem}_{i:03d}.jpg"
                cv2.imwrite(str(rec_img_dir / crop_name), crop)
                rec_lines[split_name].append(f"images/{crop_name}\t{text}")

            det_line = f"images/{img_path.name}\t{json.dumps(det_annots, ensure_ascii=False)}"
            det_lines[split_name].append(det_line)

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    (out_path / "det" / "train_det.txt").write_text("\n".join(det_lines["train"]), encoding="utf-8")
    (out_path / "det" / "val_det.txt").write_text("\n".join(det_lines["val"]), encoding="utf-8")
    (out_path / "rec" / "rec_gt_train.txt").write_text("\n".join(rec_lines["train"]), encoding="utf-8")
    (out_path / "rec" / "rec_gt_val.txt").write_text("\n".join(rec_lines["val"]), encoding="utf-8")

    sorted_chars = sorted(charset)
    (dict_dir / "word_dict.txt").write_text("\n".join(sorted_chars), encoding="utf-8")

    print(f"Train: {len(train_pairs)} ảnh, {len(rec_lines['train'])} box")
    print(f"Val:   {len(val_pairs)} ảnh, {len(rec_lines['val'])} box")
    print(f"Dict:  {len(sorted_chars)} ký tự duy nhất -> {dict_dir/'word_dict.txt'}")
    print(f"Xong. Output tại: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Folder ảnh+json -> dataset PaddleOCR (det+rec)")
    parser.add_argument("--folder", required=True, help="Folder chứa ảnh + json labelme")
    parser.add_argument("--out", required=True, help="Folder output dataset PaddleOCR")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Tỉ lệ ảnh dùng cho val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(args.folder, args.out, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()