#!/usr/bin/python3


from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel
import os
import cv2

# cargar modelo
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/home/martinez/flower_phenotyping/data/annotations/YOLO/runs/train_plant_seg/weights/best.pt",
    confidence_threshold=0.4,
)

# parámetros
min_area = 4000
input_dir = "/home/martinez/flower_phenotyping/Series04"
output_dir = "outputs/Series04_results"
os.makedirs(output_dir, exist_ok=True)

# procesar imágenes una a una
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=960,
        slice_width=960,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.2,
        postprocess_class_agnostic=False,
    )

    # 🔥 FILTRO por área
    result.object_prediction_list = [
        pred for pred in result.object_prediction_list
        if pred.bbox.area >= min_area
    ]

    # exportar visualización
    base_name = os.path.splitext(img_name)[0]
    result.export_visuals(export_dir=output_dir,file_name=f"{base_name}_pred")