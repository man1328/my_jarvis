import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. Advanced Augmentation (The 'Identity Crisis' Cure)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.1),  # Force it to look at SHAPE, not just color
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# 2. Setup ResNet-18
model = models.resnet18(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = False  # Start frozen

model.fc = nn.Linear(model.fc.in_features, 10)
device = torch.device("cpu")  # Stick to CPU for stability on your GTX 970
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 3. Training with a Mid-Way Unfreeze
print("🚀 Starting Hybrid Training...")

for epoch in range(15):
    # UNFREEZE MOMENT: At epoch 6, we let the model adjust its 'vision'
    if epoch == 5:
        print("\n🔓 UNFREEZING: Allowing the model to refine its 'eyes'...")
        for name, param in model.named_parameters():
            if "layer4" in name or "layer3" in name:
                param.requires_grad = True
        # Drop LR significantly to avoid destroying the pre-trained wisdom
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/15 - Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), 'resnet_cifar_pro.pth')
print("✅ Pro-Model Trained and Saved!")