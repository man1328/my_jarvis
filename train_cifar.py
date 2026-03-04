import torch
import torch.optim as optim
from torchvision import datasets, transforms
from cifar_model import ColorCNN

# 1. Color Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Animals look the same flipped!
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize 3 channels
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = ColorCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()

print("🏁 Starting Color Training (CIFAR-10)...")

for epoch in range(10):  # Let's start with 10
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), 'cifar_model.pth')
print("✅ Color Brain Saved!")