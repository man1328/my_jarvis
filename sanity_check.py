import cv2
import os
import random

# Update these to your actual unzipped export paths
IMAGE_DIR = "cigar_project/images"
LABEL_DIR = "cigar_project/labels"


def draw_labels():
    # Pick 3 random images to check
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_images = random.sample(image_files, 3)

    for img_name in sample_images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.rsplit('.', 1)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"⚠️ Missing label for {img_name}")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            for line in f.readlines():
                # YOLO format: class_id x_center y_center width height (normalized 0-1)
                cls, x, y, bw, bh = map(float, line.split())

                # Convert back to pixel coordinates
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange box
                cv2.putText(img, "Cigar", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.imshow("Sanity Check - Press any key", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    draw_labels()