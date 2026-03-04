import os

os.environ['POLARS_SKIP_CPU_CHECK'] = '1'

from ultralytics import YOLO

# 1. Switch to the Detection model (yolov8n.pt)
# This matches your bounding box labels!
model = YOLO('yolov8n.pt')


def train_custom_eye():
    print("🔥 Compatibility Mode: Training Detection Eye for Cigars...")

    # We keep 'workers=0' for stability on the FX-8350
    model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device='cpu',
        workers=0,
        name='jarvis_cigar_model',
        plots=True
    )

    print("✅ Success! Model saved in runs/detect/jarvis_cigar_model/weights/best.pt")


if __name__ == "__main__":
    train_custom_eye()