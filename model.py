import torch
import torch.nn as nn


# This class is our "Architectural Blueprint"
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Input Layer to Hidden Layer 1
        # We take 784 inputs and condense them to 128 features
        self.layer1 = nn.Linear(784, 128)

        # 2. Hidden Layer 1 to Hidden Layer 2
        self.layer2 = nn.Linear(128, 64)

        # 3. Hidden Layer 2 to Output
        self.output_layer = nn.Linear(64, 10)

        # 4. The Activation Function (The "Spark")
        self.relu = nn.ReLU()

        # 5. The Probability Function (The "Decision")
        self.softmax = nn.LogSoftmax(dim=1)

    # This function defines how data flows through the brain
    def forward(self, x):
        x = self.relu(self.layer1(x))  # Pass through layer 1, then activate
        x = self.relu(self.layer2(x))  # Pass through layer 2, then activate
        x = self.softmax(self.output_layer(x))  # Output final probabilities
        return x


# Let's verify it works!
if __name__ == "__main__":
    model = NeuralNet()
    print("✨ Your AI Architecture is ready:")
    print(model)