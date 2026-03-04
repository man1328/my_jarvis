import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# 1. Setup the Pro Brain
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('resnet_cifar_pro.pth', map_location='cpu'))
model.eval()

# 2. Precision Processing (Must match training!)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def run_batch(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return

    results = []
    stats = {cls: 0 for cls in classes}
    stats['Low Confidence'] = 0

    print(f"🔎 Scanning folder: {folder_path}...\n")
    print(f"{'FILENAME':<25} | {'PREDICTION':<15} | {'CONFIDENCE'}")
    print("-" * 55)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        batch = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(batch)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)

        label = classes[idx]
        confidence = conf.item()

        # Track stats
        stats[label] += 1
        if confidence < 0.50:
            stats['Low Confidence'] += 1
            status = f"⚠️ {label} (Unsure)"
        else:
            status = f"✅ {label}"

        print(f"{filename[:23]:<25} | {status:<15} | {confidence:.2%}")

    # 3. The Scorecard Summary
    print("\n" + "=" * 30)
    print("       AI SCORECARD")
    print("=" * 30)
    for cls, count in stats.items():
        if count > 0 and cls != 'Low Confidence':
            print(f"{cls:<12}: {count} found")
    print("-" * 30)
    print(f"Total Images: {len(files)}")
    print(f"Low Confidence: {stats['Low Confidence']}")
    print("=" * 30)


if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "my_test_images"
    run_batch(target_folder)