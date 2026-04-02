#!/usr/bin/python3

import ultralytics
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="/home/martinez/flower_phenotyping/data/runs/segment/train/weights/best.pt",
    model_confidence_threshold=0.3,
    source="/home/martinez/flower_phenotyping/Series04",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
    )


from sahi.postprocess.combine import NMMPostprocess

result = get_sliced_prediction(
    image="imagen.jpg",
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="NMM",          # Non-Maximum Merging (más conservador que NMS)
    postprocess_match_metric="IOS",  # Intersection over Smaller area
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False, # respetar clases al fusionar
    export_dir="outputs/",
    export_visual=True
)
