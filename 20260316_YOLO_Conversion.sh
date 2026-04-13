#!/usr/bin/env bash
# =============================================================================
# YOLO Segmentation Training Pipeline
# Converts per-image JSON annotations → COCO → SAHI slices → YOLO format
# =============================================================================
set -euo pipefail

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE="/home/martinez/flower_phenotyping/data"
WORK="${BASE}/annotations/YOLO"

RAW_ORI="${BASE}/raw"
JSON_ORI="${BASE}/annotations/json"

RAW="${WORK}/raw_data"
COCO="${WORK}/coco_out"
SAHI="${WORK}/coco_sliced"
YOLO="${WORK}/yolo_dataset"

COCO_SCRIPT="/home/martinez/flower_phenotyping/src/20260316_per_image_jsons_to_coco.py"
SAHI_SCRIPT="/home/martinez/flower_phenotyping/src/20260402_YOLO_SAHI_Training.py"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ─── CLEAN ───────────────────────────────────────────────────────────────────
log "Cleaning workspace: ${WORK}"
rm -rf "${WORK}"
mkdir -p "${RAW}" "${COCO}" "${YOLO}"

# ─── COPY DATA ───────────────────────────────────────────────────────────────
log "Copying images..."
# Copy images (gracefully skip if none found)
find "${RAW_ORI}" -maxdepth 1 \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) \
    -exec cp -t "${RAW}" {} + 2>/dev/null || log "Warning: no images found in ${RAW_ORI}"

log "Copying JSON annotations..."
find "${JSON_ORI}" -maxdepth 1 -name "*.json" \
    -exec cp -t "${RAW}" {} + 2>/dev/null || die "No JSON annotation files found in ${JSON_ORI}"

# ─── COCO CONVERSION ─────────────────────────────────────────────────────────
log "Converting to COCO format..."
python "${COCO_SCRIPT}" \
    --input "${RAW}" \
    --out   "${COCO}" \
    --val   0.1 \
    --seed  42

# ─── CLEAN COCO (remove invalid annotations) ─────────────────────────────────
log "Cleaning COCO annotations..."

# Write clean_coco.py inline so it's self-contained
cat > /tmp/clean_coco.py << 'PYSCRIPT'
#!/usr/bin/env python3
"""
Remove annotations with invalid/empty segmentation or zero-area bboxes.
Also removes images that end up with zero annotations (optional).
"""
import json
import logging
import math
import sys
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# Minimum real polygon area in pixels² to keep an annotation.
# SAHI uses min_area_ratio (default 0.1) × annotation.area — if area rounds to 0
# after float math the division explodes. A threshold of 1.0 is safe.
MIN_AREA_PX2 = 1.0
 
 
def shoelace_area(flat_poly: list[float]) -> float:
    """
    Compute polygon area using the Shoelace formula.
    Input: flat list [x0, y0, x1, y1, ...] with at least 3 points (6 values).
    Returns the absolute area (always >= 0).
    """
    if len(flat_poly) < 6:
        return 0.0
    xs = flat_poly[0::2]
    ys = flat_poly[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0
 
 
def is_valid_annotation(ann: dict, min_area: float = MIN_AREA_PX2) -> tuple[bool, str]:
    """
    Return (True, "") if the annotation is valid, or (False, reason) if it should be removed.
    Checks are ordered from cheapest to most expensive.
    """
    # 1. Segmentation must exist and be a non-empty list of rings
    seg = ann.get("segmentation")
    if not seg or not isinstance(seg, list):
        return False, "missing or invalid segmentation"
 
    # 2. Stored bbox must be sane
    bbox = ann.get("bbox", [])
    if len(bbox) < 4:
        return False, "missing bbox"
    if bbox[2] <= 0 or bbox[3] <= 0:
        return False, f"zero/negative bbox dims ({bbox[2]:.2f} × {bbox[3]:.2f})"
 
    # 3. Every ring in the segmentation must have enough points and real area
    total_real_area = 0.0
    for ring in seg:
        if not isinstance(ring, list) or len(ring) < 6:
            return False, f"ring too short ({len(ring)} values, need ≥ 6)"
        # All values must be finite numbers
        if not all(math.isfinite(v) for v in ring):
            return False, "ring contains NaN or Inf"
        total_real_area += shoelace_area(ring)
 
    # 4. Real area must be above the minimum threshold
    if total_real_area < min_area:
        return False, f"real polygon area too small ({total_real_area:.4f} < {min_area})"
 
    return True, ""
 
 
def clean(
    in_path: str,
    out_path: str,
    min_area: float = MIN_AREA_PX2,
    remove_empty_images: bool = False,
) -> None:
    log.info("Reading: %s", in_path)
    with open(in_path, encoding="utf-8") as f:
        coco = json.load(f)
 
    original_anns = coco.get("annotations", [])
    log.info("Input: %d annotations, %d images", len(original_anns), len(coco.get("images", [])))
 
    reasons: dict[str, int] = {}
    clean_anns = []
 
    for ann in original_anns:
        ok, reason = is_valid_annotation(ann, min_area)
        if ok:
            clean_anns.append(ann)
            # Also overwrite the stored area with the real computed value
            # so SAHI's division uses the correct number
            flat = ann["segmentation"][0]
            ann["area"] = shoelace_area(flat)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
 
    removed = len(original_anns) - len(clean_anns)
    log.info("Removed %d annotations:", removed)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        log.info("  • %s: %d", reason, count)
 
    coco["annotations"] = clean_anns
 
    if remove_empty_images:
        valid_img_ids = {a["image_id"] for a in clean_anns}
        before = len(coco.get("images", []))
        coco["images"] = [i for i in coco["images"] if i["id"] in valid_img_ids]
        dropped = before - len(coco["images"])
        if dropped:
            log.info("Removed %d images that had no remaining annotations", dropped)
 
    log.info(
        "Output: %d annotations, %d images",
        len(coco["annotations"]),
        len(coco.get("images", [])),
    )
 
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    log.info("Saved: %s", out_path)
 
 
def main() -> None:
    import argparse
 
    ap = argparse.ArgumentParser(
        description="Remove degenerate COCO annotations that cause SAHI ZeroDivisionError."
    )
    ap.add_argument("input",  help="Input COCO JSON path")
    ap.add_argument("output", help="Output cleaned COCO JSON path")
    ap.add_argument(
        "--min-area",
        type=float,
        default=MIN_AREA_PX2,
        help=f"Minimum real polygon area in px² to keep (default: {MIN_AREA_PX2})",
    )
    ap.add_argument(
        "--remove-empty-images",
        action="store_true",
        help="Also remove images that have no annotations after cleaning",
    )
    args = ap.parse_args()
    clean(args.input, args.output, args.min_area, args.remove_empty_images)
 
 
if __name__ == "__main__":
    main()
PYSCRIPT

python /tmp/clean_coco.py \
    "${COCO}/instances_train.json" \
    "${COCO}/instances_train_clean.json"

python /tmp/clean_coco.py \
    "${COCO}/instances_val.json" \
    "${COCO}/instances_val_clean.json"

# ─── SAHI SLICING ────────────────────────────────────────────────────────────
log "Running SAHI slicing..."
mkdir -p "${SAHI}/train" "${SAHI}/val"
python "${SAHI_SCRIPT}" --base "${WORK}"

# ─── FIX JSON NAMES (SAHI sometimes outputs a timestamped filename) ──────────
# NOTE: SAHI may generate exactly one .json per split. We rename it to instances.json
# using a loop so we handle any filename SAHI decides to output.
log "Normalising SAHI output JSON names..."
for split in train val; do
    split_dir="${SAHI}/${split}"
    json_files=( "${split_dir}"/*.json )

    if [[ ${#json_files[@]} -eq 0 ]]; then
        die "No JSON found in ${split_dir} after SAHI slicing"
    fi

    if [[ ${#json_files[@]} -gt 1 ]]; then
        # Merge multiple JSONs (edge case with SAHI versions that split output)
        log "Warning: ${#json_files[@]} JSONs in ${split_dir}, merging..."
        python - "${split_dir}" << 'PY'
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
files = sorted(d.glob("*.json"))

merged = None
for f in files:
    data = json.loads(f.read_text())
    if merged is None:
        merged = data
    else:
        max_id = max((i["id"] for i in merged["images"]), default=0)
        max_ann = max((a["id"] for a in merged["annotations"]), default=0)
        for img in data["images"]:
            img["id"] += max_id
        for ann in data["annotations"]:
            ann["id"] += max_ann
            ann["image_id"] += max_id
        merged["images"].extend(data["images"])
        merged["annotations"].extend(data["annotations"])
    f.unlink()

(d / "instances.json").write_text(json.dumps(merged, indent=2))
print(f"Merged {len(files)} files into instances.json")
PY
    else
        # Single file — just rename if necessary
        src="${json_files[0]}"
        dst="${split_dir}/instances.json"
        [[ "${src}" != "${dst}" ]] && mv "${src}" "${dst}"
    fi
done

# ─── CONVERT COCO → YOLO ─────────────────────────────────────────────────────
log "Converting COCO slices to YOLO format..."
rm -rf "${YOLO}"
mkdir -p "${YOLO}"

python - << PY
from ultralytics.data.converter import convert_coco

sahi   = "${SAHI}"
yolo   = "${YOLO}"

convert_coco(labels_dir=f"{sahi}/train", save_dir=f"{yolo}_train_tmp", use_segments=True)
convert_coco(labels_dir=f"{sahi}/val",   save_dir=f"{yolo}_val_tmp",   use_segments=True)
PY

# ─── MERGE TRAIN/VAL INTO FINAL YOLO STRUCTURE ───────────────────────────────
log "Merging into final YOLO dataset structure..."
mkdir -p "${YOLO}/images/train" "${YOLO}/images/val"
mkdir -p "${YOLO}/labels/train" "${YOLO}/labels/val"

# Use rsync for safer copy (handles missing sources gracefully)
for split in train val; do
    tmp="${YOLO}_${split}_tmp"
    # Images — ultralytics may place them under images/ or directly
    if   [[ -d "${tmp}/images" ]];        then cp -r "${tmp}/images/." "${YOLO}/images/${split}/"
    elif [[ -d "${tmp}" ]];               then find "${tmp}" -name "*.jpg" -o -name "*.png" | xargs -I{} cp {} "${YOLO}/images/${split}/" 2>/dev/null || true
    fi
    # Labels
    if   [[ -d "${tmp}/labels" ]];        then cp -r "${tmp}/labels/." "${YOLO}/labels/${split}/"
    elif [[ -d "${tmp}" ]];               then find "${tmp}" -name "*.txt" | xargs -I{} cp {} "${YOLO}/labels/${split}/" 2>/dev/null || true
    fi
    rm -rf "${tmp}"
done

# ─── GENERATE data.yaml ──────────────────────────────────────────────────────
log "Writing data.yaml..."
# Automatically extract category names from the cleaned train COCO JSON
python - << PY
import json
from pathlib import Path

coco_path = "${COCO}/instances_train_clean.json"
yaml_path  = "${WORK}/data.yaml"
yolo_path  = "${YOLO}"

with open(coco_path) as f:
    cats = json.load(f)["categories"]

# Sort by id to preserve YOLO class index order
cats = sorted(cats, key=lambda c: c["id"])
names_block = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))

yaml_content = f"""path: {yolo_path}
train: images/train
val:   images/val

nc: {len(cats)}
names:
{names_block}
"""

Path(yaml_path).write_text(yaml_content)
print(f"data.yaml written with {len(cats)} class(es):")
for i, c in enumerate(cats):
    print(f"  {i}: {c['name']}")
PY

# ─── SANITY CHECK ────────────────────────────────────────────────────────────
log "Sanity check..."
n_train_imgs=$(find "${YOLO}/images/train" -type f | wc -l)
n_train_lbls=$(find "${YOLO}/labels/train" -type f | wc -l)
n_val_imgs=$(find   "${YOLO}/images/val"   -type f | wc -l)
n_val_lbls=$(find   "${YOLO}/labels/val"   -type f | wc -l)

log "Train: ${n_train_imgs} images, ${n_train_lbls} labels"
log "Val:   ${n_val_imgs} images, ${n_val_lbls} labels"

[[ ${n_train_imgs} -eq 0 ]] && die "No training images found!"
[[ ${n_train_lbls} -eq 0 ]] && die "No training labels found!"
[[ ${n_val_imgs}   -eq 0 ]] && die "No validation images found!"

# ─── TRAIN ───────────────────────────────────────────────────────────────────
log "Starting YOLO training..."
yolo task=segment mode=train \
    model=yolo11s-seg.pt \
    data="${WORK}/data.yaml" \
    imgsz=640 \
    epochs=100 \
    batch=2 \
    project="${WORK}/runs" \
    name="train_clean"

log "✅ Pipeline complete."