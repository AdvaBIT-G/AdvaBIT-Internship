#!/usr/bin/python3

import os
import json
import glob

# Folders
ANNOTATIONS_DIR = "/home/martinez/flower_phenotyping/data/annotations/json"
OUTPUT_FILE = "instances_coco.json"

images = []
annotations = []
categories = []
category_map = {}

annotation_id = 1
image_id = 1

def convert_segmentation(points):
    """Converts [[x,y],[x,y]] -> [x1,y1,x2,y2,...]"""
    flat = []
    for x, y in points:
        flat.extend([x, y])
    return flat


def compute_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


json_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    info = data.get("info", {})
    objects = data.get("objects", [])

    file_name = info.get("name")
    width = info.get("width")
    height = info.get("height")

    images.append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    })

    for obj in objects:

        category_name = obj["category"]

        # Crear categoría si no existe
        if category_name not in category_map:
            category_id = len(category_map) + 1
            category_map[category_name] = category_id

            categories.append({
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            })

        category_id = category_map[category_name]

        points = obj["segmentation"]
        segmentation = [convert_segmentation(points)]
        bbox = compute_bbox(points)
        area = bbox[2] * bbox[3]

        # Calcular área si no existe
        area = obj.get("area", bbox[2] * bbox[3] if bbox else 0)

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(coco_output, f, indent=4)

print("✅ Conversion completed")
print(f"Imágenes: {len(images)}")
print(f"Anotaciones: {len(annotations)}")
print(f"Categorías: {len(categories)}")
print(f"Archivo generado: {OUTPUT_FILE}")