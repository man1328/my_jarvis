#!/usr/bin/venv python3
import torch
import matplotlib.pyplot as plt
from cnn_model import CNN

# 1. Load the Model
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth'))

# 2. Extract the filters from the first layer
# model.conv1.weight has shape [10, 1, 5, 5]
# (10 filters, 1 channel deep, 5 pixels tall, 5 pixels wide)
filters = model.conv1.weight.data

# 3. Normalize them (make them look nice for the plot)
# This scales the values between 0 and 1 so they show up as grayscale
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# 4. Plot them!
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
print("🔍 Visualizing the 10 Filters of Layer 1:")

for i, ax in enumerate(axes.flat):
    # Get the i-th filter
    kernel = filters[i][0]

    # Plot it
    ax.imshow(kernel, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Filter {i+1}")

plt.show()
