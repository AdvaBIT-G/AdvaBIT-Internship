#!/usr/bin/python3
import os
import json
import glob
import argparse
import random
import math
import shutil
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# ------------------------
# ARGUMENTS
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--images", required=True)
parser.add_argument("--val", type=float, default=0.1)
parser.add_argument("--tolerance", type=float, default=2.0)
args = parser.parse_args()

ANNOTATIONS_DIR = args.input
OUTPUT_DIR = args.out
IMAGES_DIR = args.images
SIMPLIFY_TOL = args.tolerance

# 🔧 fILTERS
MIN_AREA = 200        # Remove noise
MIN_FRAGMENT = 500    
MAX_POINTS = 100

CATEGORY_MAP = {"flower": 0, "plant": 1}

# ------------------------
# HELPERS
# ------------------------
def is_valid_coord(x, y):
    try:
        return x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
    except:
        return False

def safe_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom is None or geom.is_empty:
        return None
    return geom.buffer(0) 

def simplify_polygon(poly, tolerance, max_points):
    if poly.area < MIN_AREA:
        return None

    tol = tolerance
    simplified = poly.simplify(tol, preserve_topology=True)

    # 🔧 avoid destroying too much
    if simplified.area < 0.5 * poly.area:
        return poly

    while len(simplified.exterior.coords) > max_points and tol < tolerance * 50:
        tol *= 1.5
        simplified = poly.simplify(tol, preserve_topology=True)

    if simplified.is_empty:
        return poly

    return simplified

def extract_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []

def polygon_to_yolo_line(p, img_w, img_h, class_id):
    p = simplify_polygon(p, SIMPLIFY_TOL, MAX_POINTS)
    if p is None:
        return None

    coords = [(x, y) for x, y in p.exterior.coords if is_valid_coord(x, y)]
    if len(coords) < 3:
        return None

    normalized = []
    for x, y in coords:
        nx = max(0.0, min(1.0, round(x / img_w, 6)))
        ny = max(0.0, min(1.0, round(y / img_h, 6)))
        normalized.extend([nx, ny])

    return str(class_id) + " " + " ".join(map(str, normalized))

# ------------------------
# DIRS
# ------------------------
for split in ["train", "val"]:
    Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

# ------------------------
# PROCESS
# ------------------------
json_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json")))
print(f"Found {len(json_files)} JSONs\n")

all_entries = []
skipped = 0

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    info = data.get("info", {})
    objects = data.get("objects", [])

    width = info.get("width")
    height = info.get("height")
    file_name = info.get("name", "")

    if not width or not height:
        continue

    flowers = []
    plants = []

    for obj in objects:
        pts = obj.get("segmentation", [])
        if len(pts) < 3:
            continue

        poly = safe_polygon(Polygon(pts))
        if poly is None:
            skipped += 1
            continue

        if obj["category"].lower() == "flower":
            flowers.extend(extract_polygons(poly))
        else:
            plants.extend(extract_polygons(poly))

    flower_union = safe_polygon(unary_union(flowers)) if flowers else None

    yolo_lines = []

    # 🌱 PLANTS
    for plant in plants:
        diff = plant
        diff = safe_polygon(diff)
        

        for p in extract_polygons(diff):
            if p.area < MIN_FRAGMENT:
                continue
            line = polygon_to_yolo_line(p, width, height, 1)
            if line:
                yolo_lines.append(line)

    # 🌸 FLOWERS
    for fpoly in flowers:
        if fpoly.area < MIN_AREA:
            continue
        line = polygon_to_yolo_line(fpoly, width, height, 0)
        if line:
            yolo_lines.append(line)

    stem = Path(file_name).stem if file_name else Path(json_file).stem
    all_entries.append((json_file, stem, yolo_lines))

# ------------------------
# SPLIT
# ------------------------
random.shuffle(all_entries)
val_size = max(1, int(len(all_entries) * args.val))
val_entries = all_entries[:val_size]
train_entries = all_entries[val_size:]

def write(entries, split):
    miss = 0
    for _, stem, lines in entries:
        img = None
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(IMAGES_DIR, stem + ext)
            if os.path.exists(p):
                img = p
                break

        if img:
            shutil.copy2(img, f"{OUTPUT_DIR}/images/{split}/{os.path.basename(img)}")
        else:
            miss += 1

        with open(f"{OUTPUT_DIR}/labels/{split}/{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    return miss

print("Writing train...")
m1 = write(train_entries, "train")
print("Writing val...")
m2 = write(val_entries, "val")

print("\n✅ DONE")
print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")
print(f"⚠️ invalid polygons: {skipped}")
print(f"⚠️ images not included: {m1 + m2}")