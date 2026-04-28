from ultralytics import YOLO

model = YOLO("/home/martinez/flower_phenotyping/data/annotations/YOLO/runs/train_clean/weights/best.pt")

results = model.predict(
    source="/home/martinez/flower_phenotyping/data/annotations/YOLO/Predict_images",
    imgsz=1024,
    conf=0.3,
    save=True
)

# Mostrar resultado
results[0].show()
