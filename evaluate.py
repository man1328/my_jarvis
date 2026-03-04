import torch
import matplotlib.pyplot as plt
from cnn_model import CNN
from torchvision import datasets, transforms

# 1. Define the Fashion Dictionary 📖
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 2. Load Data (Fashion-MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. Load Model
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# 4. Calculate Accuracy
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"🎯 Accuracy on Fashion Items: {100 * correct / total:.2f}%")

# --- 5. VISUALIZE PREDICTIONS ---
# Let's see what the model is getting right (or wrong)
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(10, 4))
for i in range(10):  # Show 10 images
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    # Un-normalize image to show it clearly
    img = images[i] / 2 + 0.5
    img = img.numpy().transpose((1, 2, 0)) # Move channels to the end
    plt.imshow(img.squeeze(), cmap='gray') # Grayscale

    # Check if correct
    is_correct = predicted[i] == labels[i]
    color = 'green' if is_correct else 'red'

    # Title: "Pred (Actual)"
    ax.set_title(f"{classes[predicted[i]]}\n({classes[labels[i]]})", color=color, fontsize=8)

plt.tight_layout()
plt.show()
