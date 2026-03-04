import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the Transformation
# Raw images are just pixel files. We need to:
# A. Convert them to Tensors (math-friendly numbers)
# B. Normalize them (squish values to be between -1 and 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Download the "Textbooks" (MNIST Data)
print("Downloading data... (this might take a moment)")
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# 3. Build the Conveyor Belt (DataLoader)
# batch_size=64 means we feed 64 images at a time
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

print(f"✅ Data Ready! We have {len(trainset)} images for training.")