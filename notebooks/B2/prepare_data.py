"""
Detect text bằng PaddleOCR (chỉ dùng module detection),
Recog text bằng VietOCR (đọc tốt tiếng Việt có dấu),
Xuất mỗi ảnh 1 file .json dạng labelme/PPOCRLabel, lưu ngay trong folder ảnh.

Cài đặt cần thiết:
    pip install paddleocr paddlepaddle vietocr opencv-python pillow

Usage:
    python ocr_to_labelme.py --folder /path/to/images --score-thresh 0.5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from paddleocr import TextDetection
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LABEL_DEFAULT = "text"
LABELME_VERSION = "0.4.36"


def build_detector() -> TextDetection:
    # Module detection thuần của PaddleOCR (không recog)
    return TextDetection()


def build_recognizer(device: str) -> Predictor:
    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = device  # "cpu" hoặc "cuda:0"
    config["predictor"]["beamsearch"] = False
    return Predictor(config)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự tl, tr, br, bl để warp đúng chiều."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_quad(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crop + warp 1 vùng tứ giác (poly 4 điểm) thành ảnh chữ nhật thẳng."""
    rect = order_points(poly.astype("float32"))
    (tl, tr, br, bl) = rect

    w_top = np.linalg.norm(tr - tl)
    w_bottom = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)

    max_w = max(int(w_top), int(w_bottom), 1)
    max_h = max(int(h_left), int(h_right), 1)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


def recognize_texts(image_bgr: np.ndarray, dt_polys, recognizer: Predictor):
    texts = []
    for poly in dt_polys:
        poly = np.array(poly)
        crop = crop_quad(image_bgr, poly)
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            texts.append("")
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        try:
            text = recognizer.predict(pil_crop)
        except Exception:
            text = ""
        texts.append(text)
    return texts


def build_labelme_json(img_path: Path, dt_polys, texts) -> dict:
    with Image.open(img_path) as im:
        w, h = im.size

    shapes = []
    for poly, text in zip(dt_polys, texts):
        points = [[float(x), float(y)] for x, y in poly]
        shapes.append(
            {
                "label": LABEL_DEFAULT,
                "text": text,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )

    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def process_folder(folder: str, score_thresh: float, device: str, overwrite: bool):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Không tìm thấy folder: {folder}")

    img_paths = sorted(
        p for p in folder_path.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not img_paths:
        print("Không có ảnh nào trong folder.")
        return

    print(f"Tìm thấy {len(img_paths)} ảnh. Đang khởi tạo detector (PaddleOCR) + recognizer (VietOCR)...")
    detector = build_detector()
    recognizer = build_recognizer(device)

    ok, skipped, failed = 0, 0, 0
    for i, img_path in enumerate(img_paths, 1):
        out_path = img_path.with_suffix(".json")
        if out_path.exists() and not overwrite:
            print(f"[{i}/{len(img_paths)}] Bỏ qua (đã có json): {img_path.name}")
            skipped += 1
            continue

        try:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                raise ValueError("Không đọc được ảnh (file lỗi hoặc không hỗ trợ)")

            det_result = detector.predict(str(img_path))[0]
            dt_polys = det_result["dt_polys"]
            dt_scores = det_result.get("dt_scores", [1.0] * len(dt_polys))

            filtered_polys = [
                p for p, s in zip(dt_polys, dt_scores) if s >= score_thresh
            ]

            texts = recognize_texts(image_bgr, filtered_polys, recognizer)
            data = build_labelme_json(img_path, filtered_polys, texts)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[{i}/{len(img_paths)}] OK: {img_path.name} -> {out_path.name} "
                  f"({len(data['shapes'])} shapes)")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(img_paths)}] LỖI với {img_path.name}: {e}")
            failed += 1

    print(f"\nHoàn tất. Thành công: {ok}, bỏ qua: {skipped}, lỗi: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect (PaddleOCR) + Recog (VietOCR) -> json labelme"
    )
    parser.add_argument("--folder", required=True, help="Đường dẫn folder chứa ảnh")
    parser.add_argument("--score-thresh", type=float, default=0.5,
                         help="Ngưỡng loại box detect có score thấp")
    parser.add_argument("--device", default="cpu",
                         help="cpu hoặc cuda:0 (cho VietOCR)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Ghi đè json đã tồn tại")
    args = parser.parse_args()

    process_folder(args.folder, args.score_thresh, args.device, args.overwrite)


if __name__ == "__main__":
    main()