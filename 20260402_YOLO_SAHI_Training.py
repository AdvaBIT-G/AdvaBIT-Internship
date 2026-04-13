#!/usr/bin/env python3
"""
Slice COCO-format annotations and images using SAHI for YOLO training.
"""
 
import argparse
import logging
import sys
from pathlib import Path
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
def validate_paths(coco_file: str, image_dir: str) -> None:
    """Check that required input files/dirs exist before processing."""
    if not Path(coco_file).is_file():
        log.error("COCO annotation file not found: %s", coco_file)
        sys.exit(1)
    if not Path(image_dir).is_dir():
        log.error("Image directory not found: %s", image_dir)
        sys.exit(1)
 
 
def run_slice(
    coco_file: str,
    image_dir: str,
    output_dir: str,
    slice_h: int,
    slice_w: int,
    overlap_h: float,
    overlap_w: float,
    ignore_negatives: bool,
) -> None:
    """Run SAHI slicing for a single split."""
    try:
        from sahi.slicing import slice_coco
    except ImportError:
        log.error("SAHI is not installed. Run: pip install sahi")
        sys.exit(1)
 
    validate_paths(coco_file, image_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    log.info("Slicing: %s → %s", coco_file, output_dir)
 
    result = slice_coco(
        coco_annotation_file_path=coco_file,
        image_dir=image_dir,
        output_dir=output_dir,
        output_coco_annotation_file_name="instances.json",
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
        ignore_negative_samples=ignore_negatives,
    )
 
    log.info("Done: %s", output_dir)
    return result
 
 
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Slice COCO annotations with SAHI for YOLO training."
    )
    ap.add_argument(
        "--base",
        default="/home/martinez/flower_phenotyping/data/annotations/YOLO",
        help="Base directory for annotations",
    )
    ap.add_argument("--slice-height",  type=int,   default=1024, help="Slice height in pixels")
    ap.add_argument("--slice-width",   type=int,   default=1024, help="Slice width in pixels")
    ap.add_argument("--overlap-h",     type=float, default=0.2,  help="Vertical overlap ratio")
    ap.add_argument("--overlap-w",     type=float, default=0.2,  help="Horizontal overlap ratio")
    ap.add_argument("--keep-negatives", action="store_true",     help="Keep slices with no annotations")
    args = ap.parse_args()
 
    BASE      = args.base
    coco_dir  = f"{BASE}/coco_out"
    raw_dir   = f"{BASE}/raw_data"
    sahi_dir  = f"{BASE}/coco_sliced"
 
    splits = [
        ("train", f"{coco_dir}/instances_train_clean.json", f"{sahi_dir}/train"),
        ("val",   f"{coco_dir}/instances_val_clean.json",   f"{sahi_dir}/val"),
    ]
 
    for split_name, coco_file, out_dir in splits:
        log.info("=== Processing split: %s ===", split_name)
        run_slice(
            coco_file=coco_file,
            image_dir=raw_dir,
            output_dir=out_dir,
            slice_h=args.slice_height,
            slice_w=args.slice_width,
            overlap_h=args.overlap_h,
            overlap_w=args.overlap_w,
            ignore_negatives=not args.keep_negatives,
        )
 
    log.info("All splits processed.")
 
 
if __name__ == "__main__":
    main()