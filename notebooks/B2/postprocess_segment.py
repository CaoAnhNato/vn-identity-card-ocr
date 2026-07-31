from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


# ============================================================
# HÀM PHỤ: làm đen nền ngoài polygon
# ============================================================
def apply_black_background(orig_img, poly):
    poly_int = poly.astype(np.int32)
    mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)
    return cv2.bitwise_and(orig_img, orig_img, mask=mask)


# ============================================================
# HÀM PHỤ: xoay theo góc nghiêng + crop theo bounding box
# ============================================================
def rotate_and_crop(img_with_black_bg, poly):
    poly_int = poly.astype(np.int32)
    rect = cv2.minAreaRect(poly)
    angle = rect[2]

    h_img, w_img = img_with_black_bg.shape[:2]
    center = (w_img // 2, h_img // 2)

    if angle < -45:
        angle += 90

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_with_black_bg, M, (w_img, h_img),
                              flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

    # xoay polygon theo cùng ma trận để tính lại bounding box đúng vị trí mới
    ones = np.ones((poly_int.shape[0], 1))
    poly_hom = np.hstack([poly_int, ones])
    rotated_poly = (M @ poly_hom.T).T.astype(np.int32)

    x, y, w, h = cv2.boundingRect(rotated_poly)
    x, y = max(0, x), max(0, y)
    if w <= 0 or h <= 0:
        return None
    return rotated[y:y+h, x:x+w]


# ============================================================
# HÀM CHÍNH: chạy full train/valid/test -> lưu chung 1 folder
# ============================================================
def process_all_splits(
    model_path,
    dataset_root,
    output_dir,
    splits=("train", "valid", "test"),
    images_subdir="images",
    conf=0.25,
    img_exts=(".jpg", ".jpeg", ".png"),
):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    log_no_detect = []
    log_error = []
    total_saved = 0

    for split in splits:
        split_dir = os.path.join(dataset_root, split, images_subdir)
        if not os.path.isdir(split_dir):
            print(f"[SKIP] Không tìm thấy folder: {split_dir}")
            continue

        img_paths = [
            p for p in glob(os.path.join(split_dir, "*"))
            if os.path.splitext(p)[1].lower() in img_exts
        ]
        print(f"\n[{split}] Tìm thấy {len(img_paths)} ảnh")

        for img_path in tqdm(img_paths, desc=f"Processing {split}"):
            try:
                results = model.predict(img_path, conf=conf, verbose=False)
                r = results[0]
                orig_img = r.orig_img
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                if r.masks is None or len(r.masks.xy) == 0:
                    log_no_detect.append(img_path)
                    continue

                for j, poly in enumerate(r.masks.xy):
                    black_bg_img = apply_black_background(orig_img, poly)
                    cropped = rotate_and_crop(black_bg_img, poly)

                    if cropped is None:
                        continue

                    out_name = f"{split}_{base_name}_{j}.jpg"
                    out_path = os.path.join(output_dir, out_name)
                    cv2.imwrite(out_path, cropped)
                    total_saved += 1

            except Exception as e:
                log_error.append((img_path, str(e)))

    print(f"\n{'='*50}")
    print(f"Tổng ảnh đã lưu: {total_saved}")
    print(f"Ảnh không detect được: {len(log_no_detect)}")
    print(f"Ảnh lỗi: {len(log_error)}")
    if log_no_detect:
        print("\nDanh sách ảnh không detect (10 đầu):")
        for p in log_no_detect[:10]:
            print(" -", p)
    if log_error:
        print("\nDanh sách ảnh lỗi:")
        for p, e in log_error:
            print(" -", p, "|", e)

    return {
        "total_saved": total_saved,
        "no_detect": log_no_detect,
        "errors": log_error,
    }


# ============================================================
# CHẠY
# ============================================================
report = process_all_splits(
    model_path="./runs/segment/train/weights/best.pt",
    dataset_root="./Segmentation_Datasets",
    output_dir="./warped_bills",
    splits=("train", "valid", "test"),
)