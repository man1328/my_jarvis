from ultralytics import YOLO
import cv2

# 1. Load the model (Nano version)
model = YOLO('yolov8n.pt')


def monitor_room(source_path):
    # Run the eye with device='cpu' to keep your GTX 970 happy
    # We use stream=True for faster processing
    results = model.predict(source=source_path, show=True, device='cpu', stream=True)

    print("🛡️ Pet Monitor Active. Press 'q' on the image window to stop.")

    for result in results:
        # Get labels of everything found in the current frame
        found_objects = [model.names[int(box.cls[0])] for box in result.boxes]

        if 'dog' in found_objects:
            print("🚨 SECURITY ALERT: Thor the Bulldog detected on premises!")

        if 'bird' in found_objects:
            print("🦜 AIR SUPPORT: Cosmo the Sun Conure has entered the chat!")

        if 'cat' in found_objects:
            print("🐾 FELINE UPDATE: Luna or Shakira detected!")


# To run on your webcam: monitor_room(0)
# To run on your folder again:
monitor_room("my_test_images")