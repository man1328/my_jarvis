import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from cifar_model import ColorCNN
import sys

# 1. Setup the Brain
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = ColorCNN()
model.load_state_dict(torch.load('cifar_model.pth'))
model.eval()

# 2. The Color Processor
# This must match EXACTLY what we did during training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def inspect_image(image_path):
    try:
        # Load and convert to RGB (removes transparency/alpha channels)
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            output = model(batch_t)
            # Convert logs to actual percentages
            probabilities = torch.exp(output)[0]

            # Get Top 3
            top_probs, top_indices = torch.topk(probabilities, 3)

        print(f"\n--- 🌈 Color Analysis for {image_path} ---")
        for i in range(3):
            name = classes[top_indices[i]]
            conf = top_probs[i].item()
            print(f"{i + 1}. {name}: {conf:.2%}")

    except Exception as e:
        print(f"Error: Could not process image. {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_image(sys.argv[1])
    else:
        print("Usage: python inspect_color.py your_image.jpg")