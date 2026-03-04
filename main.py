import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from cnn_model import CNN

# 1. DATA PREPARATION (Fashion-MNIST)
transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. INITIALIZE MODEL, LOSS, AND OPTIMIZER
model = CNN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. THE SCHEDULER (Fixed: Removed 'verbose')
# It waits 2 epochs (patience) of no improvement before cutting LR in half
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 4. TRAINING LOOP
print("🚀 Starting the Final Push for 90%+ Accuracy...")
epochs = 15

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # 🆕 Manually check the current learning rate to see the scheduler in action
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.6f}")

    # Update the scheduler
    scheduler.step(epoch_loss)

# 5. SAVE THE BRAIN
torch.save(model.state_dict(), 'mnist_model.pth')
print("\n🎉 Training Complete! Model saved to 'mnist_model.pth'")