#!/usr/bin/env python3
"""
Convert per-image JSON annotations to COCO format and split into train/val.
"""
 
import argparse
import json
import logging
import random
import sys
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
def load_jsons(path: str) -> list[tuple[Path, dict]]:
    """Load all valid annotation JSONs from a directory."""
    files = sorted(Path(path).glob("*.json"))
    if not files:
        log.error("No JSON files found in: %s", path)
        sys.exit(1)
 
    out = []
    skipped = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "info" in data and "objects" in data:
                out.append((f, data))
            else:
                log.debug("Skipping %s: missing 'info' or 'objects' keys", f.name)
                skipped += 1
        except json.JSONDecodeError as e:
            log.warning("Skipping %s: invalid JSON (%s)", f.name, e)
            skipped += 1
 
    log.info("Loaded %d annotation files (%d skipped)", len(out), skipped)
    return out
 
 
def build_coco(data: list[tuple[Path, dict]]) -> dict:
    """Build a COCO-format dict from per-image annotation data."""
    images, annotations, categories = [], [], []
    cat_map: dict[str, int] = {}
    ann_id = 1
    img_id = 1
    filtered_bbox = 0
    filtered_poly = 0
 
    def get_cat(name: str) -> int:
        if name not in cat_map:
            cid = len(cat_map) + 1
            cat_map[name] = cid
            categories.append({"id": cid, "name": name})
        return cat_map[name]
 
    for f, d in data:
        info = d["info"]
        fname = info.get("name", f.stem)
        img_w = info.get("width")
        img_h = info.get("height")
 
        if img_w is None or img_h is None:
            log.warning("Skipping image %s: missing width/height in info", fname)
            continue
 
        images.append({
            "id": img_id,
            "file_name": fname,
            "width": img_w,
            "height": img_h,
        })
 
        for obj in d.get("objects", []):
            seg = obj.get("segmentation", [])
 
            # Need at least 3 points for a valid polygon
            if len(seg) < 3:
                filtered_poly += 1
                continue
 
            try:
                poly = [(float(pt[0]), float(pt[1])) for pt in seg]
            except (TypeError, IndexError, ValueError) as e:
                log.debug("Skipping annotation in %s: bad segmentation (%s)", fname, e)
                filtered_poly += 1
                continue
 
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
 
            if bbox_w < 1 or bbox_h < 1:
                filtered_bbox += 1
                continue
 
            area = bbox_w * bbox_h
            if area < 1:
                filtered_bbox += 1
                continue
 
            bbox = [min(xs), min(ys), bbox_w, bbox_h]
            flat_poly = [v for xy in poly for v in xy]
            cid = get_cat(obj["category"])
 
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "segmentation": [flat_poly],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
 
        img_id += 1
 
    log.info(
        "Built COCO: %d images, %d annotations, %d categories "
        "(filtered: %d tiny bbox, %d bad polygons)",
        len(images), len(annotations), len(categories),
        filtered_bbox, filtered_poly,
    )
 
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
 
 
def split_and_save(coco: dict, out: str, val_ratio: float, seed: int = 42) -> None:
    """Split COCO dataset into train/val and save JSON files."""
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
 
    imgs = list(coco["images"])  # copy to avoid mutating in-place
    random.seed(seed)
    random.shuffle(imgs)
 
    n_val = max(1, int(len(imgs) * val_ratio)) if imgs else 0
    val_ids   = {i["id"] for i in imgs[:n_val]}
    train_ids = {i["id"] for i in imgs[n_val:]}
 
    def subset(ids: set) -> dict:
        return {
            "images":      [i for i in coco["images"]      if i["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
            "categories":  coco["categories"],
        }
 
    train_data = subset(train_ids)
    val_data   = subset(val_ids)
 
    (out_path / "instances_train.json").write_text(
        json.dumps(train_data, indent=2), encoding="utf-8"
    )
    (out_path / "instances_val.json").write_text(
        json.dumps(val_data, indent=2), encoding="utf-8"
    )
 
    log.info(
        "Saved: %d train images / %d train anns  |  %d val images / %d val anns",
        len(train_data["images"]), len(train_data["annotations"]),
        len(val_data["images"]),   len(val_data["annotations"]),
    )
    log.info("Output: %s", out_path.resolve())
 
 
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert per-image JSONs to COCO format with train/val split."
    )
    ap.add_argument("--input",  required=True,  help="Directory containing per-image JSON files")
    ap.add_argument("--out",    required=True,  help="Output directory for COCO JSONs")
    ap.add_argument("--val",    type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    ap.add_argument("--seed",   type=int,   default=42,  help="Random seed for reproducibility")
    args = ap.parse_args()
 
    if not 0 < args.val < 1:
        ap.error("--val must be between 0 and 1 (exclusive)")
 
    data = load_jsons(args.input)
    if not data:
        log.error("No valid annotation files found. Exiting.")
        sys.exit(1)
 
    coco = build_coco(data)
    split_and_save(coco, args.out, args.val, args.seed)
 
 
if __name__ == "__main__":
    main()
 