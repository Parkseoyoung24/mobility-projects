import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "dataset"          # jpg + LabelMe json
OUTPUT_DIR = BASE_DIR / "yolo_dataset"
YAML_PATH = BASE_DIR / "dataset.yaml"

CLASSES = ["red_light", "yellow_light", "green_light", "left_light"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def get_image_size(image_path, data):
    w, h = data.get("imageWidth"), data.get("imageHeight")
    if w and h:
        return int(w), int(h)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def labelme_to_yolo(image_path, json_path, txt_path):
    """Convert LabelMe boxes to YOLO lines (class xc yc w h, normalized)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    W, H = get_image_size(image_path, data)

    lines, unknown, invalid = [], [], 0
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip()
        if not label:
            continue
        if label not in CLASS_TO_ID:
            unknown.append(label)
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            invalid += 1
            continue

        # rectangle or polygon -> bounding box, clipped to the image
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = clamp(min(xs), 0, W), clamp(max(xs), 0, W)
        y1, y2 = clamp(min(ys), 0, H), clamp(max(ys), 0, H)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            invalid += 1
            continue

        xc = clamp((x1 + x2) / 2 / W, 0, 1)
        yc = clamp((y1 + y2) / 2 / H, 0, 1)
        bw = clamp(bw / W, 0, 1)
        bh = clamp(bh / H, 0, 1)
        lines.append(f"{CLASS_TO_ID[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return lines, unknown, invalid


def process_split(pairs, split):
    img_dir = OUTPUT_DIR / "images" / split
    lbl_dir = OUTPUT_DIR / "labels" / split
    stats = {"classes": Counter(), "unknown": Counter(), "boxes": 0, "invalid": 0, "empty": 0}

    for i, (img, js) in enumerate(pairs, 1):
        shutil.copy2(img, img_dir / img.name)
        lines, unknown, invalid = labelme_to_yolo(img, js, lbl_dir / f"{img.stem}.txt")

        stats["boxes"] += len(lines)
        stats["invalid"] += invalid
        stats["empty"] += len(lines) == 0
        stats["unknown"].update(unknown)
        stats["classes"].update(int(l.split()[0]) for l in lines)

        if i % 100 == 0 or i == len(pairs):
            print(f"  {split}: {i}/{len(pairs)}")
    return stats


def write_yaml():
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    YAML_PATH.write_text(
        "path: /workspace/yolo_dataset\n\n"   # docker mount path
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n{names}",
        encoding="utf-8",
    )


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1) < 1e-6, "ratios must sum to 1"
    if not INPUT_DIR.exists():
        print("dataset folder not found:", INPUT_DIR)
        return

    images = sorted(p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    pairs = [(p, p.with_suffix(".json")) for p in images if p.with_suffix(".json").exists()]
    no_json = [p.name for p in images if not p.with_suffix(".json").exists()]
    stems = {p.stem for p in images}
    no_image = [j.name for j in INPUT_DIR.glob("*.json") if j.stem not in stems]

    print(f"images: {len(images)}, pairs: {len(pairs)}")
    if no_json:
        print(f"images without json: {len(no_json)}", no_json[:20])
    if no_image:
        print(f"json without image: {len(no_image)}", no_image[:20])
    if not pairs:
        return

    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
    split_pairs = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    for s in SPLITS:
        print(f"{s}: {len(split_pairs[s])} ({len(split_pairs[s]) / n:.1%})")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for s in SPLITS:
        (OUTPUT_DIR / "images" / s).mkdir(parents=True)
        (OUTPUT_DIR / "labels" / s).mkdir(parents=True)

    results = {s: process_split(split_pairs[s], s) for s in SPLITS}
    write_yaml()

    print("\nboxes per class (train / val / test)")
    for i, name in enumerate(CLASSES):
        counts = [results[s]["classes"][i] for s in SPLITS]
        print(f"  {i} {name}: {counts[0]} / {counts[1]} / {counts[2]}  total {sum(counts)}")

    total = lambda key: sum(results[s][key] for s in SPLITS)
    print(f"total boxes: {total('boxes')}, images without boxes: {total('empty')}, invalid boxes: {total('invalid')}")

    unknown = sum((results[s]["unknown"] for s in SPLITS), Counter())
    if unknown:
        print("labels not in CLASSES (skipped):", dict(unknown))

    print("done:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
