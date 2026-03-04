import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.10) # 🆕 Reduced from 0.25

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(0.10) # 🆕 Reduced from 0.25

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout3 = nn.Dropout(0.30) # 🆕 Reduced from 0.5
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Layer 1 + Dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        # Layer 2 + Dropout
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        # Layer 3
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(-1, 128 * 3 * 3)

        # Classification + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout3(x) # 🆕 Forces variety in thinking
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
