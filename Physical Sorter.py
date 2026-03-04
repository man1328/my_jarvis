import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Setup (Same as before)
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('resnet_cifar_pro.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def sort_my_photos(source_folder):
    # Create the output directory
    output_root = "sorted_by_ai"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print(f"🚀 Sorting images from {source_folder} into {output_root}...")

    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in files:
        img_path = os.path.join(source_folder, filename)
        img = Image.open(img_path).convert('RGB')
        batch = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(batch)
            idx = torch.argmax(output, 1).item()

        label = classes[idx]

        # Create subfolder for the class
        class_folder = os.path.join(output_root, label)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Copy the file instead of moving (safer for your first run!)
        shutil.copy(img_path, os.path.join(class_folder, filename))

    print(f"✅ Finished! Check the '{output_root}' folder to see your AI's handiwork.")


if __name__ == "__main__":
    sort_my_photos("my_test_images")