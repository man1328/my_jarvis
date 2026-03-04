import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn as nn

# 1. Load the model
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('resnet_cifar_v2.pth', map_location='cpu'))
model.eval()

# 2. Load Validation Data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

all_preds = []
all_labels = []

print("📊 Analyzing the model's confusion...")
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# 3. Plot the Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Where is the model getting confused?')
plt.show()