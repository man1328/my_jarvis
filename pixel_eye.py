from ultralytics import YOLO
import cv2
import numpy as np

# 1. Load the Segmentation Model
# The '-seg' suffix tells YOLO to look for pixel boundaries
model = YOLO('yolov8n-seg.pt')


def segment_everything(source_path):
    # 2. Run Inference
    # We use device='cpu' to keep your GTX 970 stable
    results = model.predict(source=source_path, show=True, device='cpu', save=True)

    print(f"\n--- 🖋️ Instance Segmentation Report ---")

    for result in results:
        # Check if any masks were actually found
        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                class_id = int(result.boxes.cls[i])
                label = result.names[class_id]

                # Calculate the exact area in pixels!
                area = cv2.contourArea(mask.astype(np.float32))

                print(f"📍 {label.upper()} detected. Boundary covers {int(area)} pixels.")
        else:
            print("No distinct object boundaries found.")

    print(f"\n✅ Check 'runs/segment/predict' to see the 'Magic Lasso' effect!")


if __name__ == "__main__":
    # Test it on your truck or Thor!
    segment_everything("my_test_images/my_pet_photo4.jpg")