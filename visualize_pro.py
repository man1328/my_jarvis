import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 1. Load the Pro Model
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('resnet_cifar.pth', map_location='cpu'))
model.eval()

# Hook Setup
activations = {}


def get_activations(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


model.layer4.register_forward_hook(get_activations('layer4'))


def generate_heatmap(img_path):
    # 1. Load and Transform
    raw_img = Image.open(img_path).convert('RGB')
    width, height = raw_img.size

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_tensor = transform(raw_img).unsqueeze(0)

    # 2. Forward Pass
    output = model(input_tensor)
    pred_idx = torch.argmax(output).item()

    # 3. Get Activations and Weights
    act = activations['layer4']  # Shape: [1, 512, 4, 4]
    weights = model.fc.weight[pred_idx]  # Shape: [512]

    # 4. Calculate CAM using PyTorch (faster and safer)
    # We multiply each of the 512 channels by its importance weight
    cam = torch.zeros(act.shape[2:], dtype=torch.float32)
    for i in range(512):
        cam += weights[i] * act[0, i, :, :]

    # 5. Process the Heatmap
    cam = torch.relu(cam)  # Keep only positive influence

    # --- THE CRITICAL CONVERSION ---
    # We move to CPU, detach from math history, and force it to be a float32 NumPy array
    cam_np = cam.cpu().detach().numpy().astype(np.float32)

    # Normalize 0.0 to 1.0
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # 6. OpenCV Resizing (Now it will accept cam_np!)
    heatmap_resized = cv2.resize(cam_np, (width, height))

    # 7. Color and Merge
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    original_bgr = np.array(raw_img)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

    result = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    output_path = "heatmap_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"✅ Analysis complete! Predicted: {classes[pred_idx]}")
    print(f"Check '{output_path}' to see the hotspots.")


if __name__ == "__main__":
    generate_heatmap("my_test_images/truck10.jpg")  # Replace with your image name