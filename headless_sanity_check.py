import cv2
import os
import random

# Update these to your actual paths
IMAGE_DIR = "cigar_project/images"
LABEL_DIR = "cigar_project/labels"
OUTPUT_DIR = "check_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def save_sanity_checks():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_images = random.sample(image_files, min(5, len(image_files)))

    print(f"🕵️ Checking {len(sample_images)} images...")

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
                cls, x, y, bw, bh = map(float, line.split())

                # Convert normalized YOLO coordinates to pixels
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                # Draw Orange Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(img, "Cigar", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        save_path = os.path.join(OUTPUT_DIR, f"check_{img_name}")
        cv2.imwrite(save_path, img)
        print(f"✅ Saved check to: {save_path}")

    print(f"\n🚀 Done! Go to the '{OUTPUT_DIR}' folder to see your labeled cigars.")


if __name__ == "__main__":
    save_sanity_checks()