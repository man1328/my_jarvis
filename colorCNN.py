import torch
import matplotlib.pyplot as plt
from cifar_model import ColorCNN

model = ColorCNN()
model.load_state_dict(torch.load('cifar_model.pth'))

# Get the weights of the first layer (32 filters, each 3x3 with 3 color channels)
kernels = model.conv1.weight.detach()

# Normalize for display
kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

# Plot the 32 "eyes" of the model
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    # Transpose from (3, 3, 3) to (3, 3, 3) for imshow
    kernel = kernels[i].permute(1, 2, 0)
    ax.imshow(kernel)
    ax.axis('off')
    ax.set_title(f"Eye {i}")

plt.suptitle("The 32 Color-Detectors of your Model")
plt.show()