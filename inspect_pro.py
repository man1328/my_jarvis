import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# 1. Reconstruct the Pro Brain
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Load the new Pro-Model weights
model.load_state_dict(torch.load('resnet_cifar_pro.pth', map_location='cpu'))
model.eval()

# 2. Precision Processing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def inspect(img_path):
    img = Image.open(img_path).convert('RGB')
    batch = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(batch)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_conf, top_idx = torch.topk(probs, 3)

    main_conf = top_conf[0].item()
    main_label = classes[top_idx[0]]

    print(f"\n--- 🏆 Pro Analysis for {img_path} ---")

    # 🆕 The "Honesty" Filter
    if main_conf < 0.50:
        print(f"⚠️ LOW CONFIDENCE: I'm not entirely sure, but this looks most like a {main_label}.")
    else:
        print(f"✅ HIGH CONFIDENCE: This is definitely a {main_label}!")

    for i in range(3):
        print(f"{i + 1}. {classes[top_idx[i]]}: {top_conf[i]:.2%}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_pro.py your_image.jpg")