import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


CLASS_IDS = {"bg": 0, "crack": 1, "peel": 2, "disc": 3}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def preprocess_image(image_bgr: np.ndarray, target_max_dim: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image_bgr, scale


def detect_cracks(gray: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    gray_eq = cv2.equalizeHist(gray)
    grad_x = cv2.Sobel(gray_eq, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_eq, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    _, th = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(th)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 80:
            continue
        if (w > 4 * h) or (h > 4 * w) or (area < 200 and max(w, h) > 40):
            boxes.append((x, y, w, h))
            cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask


def detect_peeling(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    low_sat = cv2.inRange(hsv, (0, 0, 40), (180, 70, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(low_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(low_sat)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 400:
            continue
        boxes.append((x, y, w, h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask


def detect_discoloration(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    lower = np.array([0, 0, 180])
    upper = np.array([180, 90, 255])
    light_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(light_mask)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 300:
            continue
        boxes.append((x, y, w, h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask


def build_pseudo_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, mask_crack = detect_cracks(gray)
    _, mask_peel = detect_peeling(hsv)
    _, mask_disc = detect_discoloration(hsv)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[mask_disc > 0] = CLASS_IDS["disc"]
    mask[mask_peel > 0] = CLASS_IDS["peel"]
    mask[mask_crack > 0] = CLASS_IDS["crack"]
    return mask


def main() -> None:
    root = Path("D:/桌面桌面/Restore")
    raw_dir = root / "dataset_raw"
    out_dir = root / "dataset"
    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    splits_out = out_dir / "splits"
    ensure_dir(images_out)
    ensure_dir(masks_out)
    ensure_dir(splits_out)

    # classes expected in raw_dir: crack, peel, disc, clean
    subdirs = ["crack", "peel", "disc", "clean"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []

    for sub in subdirs:
        d = raw_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                samples.append((sub, p))

    rng = np.random.default_rng(seed=20250917)
    rng.shuffle(samples)

    train_list: List[str] = []
    val_list: List[str] = []

    for idx, (label_dir, img_path) in enumerate(samples):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_proc, _ = preprocess_image(img)
        if label_dir == "clean":
            mask = np.zeros(img_proc.shape[:2], dtype=np.uint8)
        else:
            mask = build_pseudo_mask(img_proc)

        base = f"{label_dir}_{idx:06d}"
        img_dst = images_out / f"{base}.png"
        msk_dst = masks_out / f"{base}.png"

        cv2.imencode('.png', img_proc)[1].tofile(str(img_dst))
        cv2.imencode('.png', mask)[1].tofile(str(msk_dst))

        if idx % 10 < 8:
            train_list.append(base)
        else:
            val_list.append(base)

    with open(splits_out / "train.txt", "w", encoding="utf-8") as f:
        for b in train_list:
            f.write(b + "\n")
    with open(splits_out / "val.txt", "w", encoding="utf-8") as f:
        for b in val_list:
            f.write(b + "\n")

    print(f"Prepared {len(train_list)} train and {len(val_list)} val samples at {out_dir}")


if __name__ == "__main__":
    main()


