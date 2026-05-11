# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Multi-Layer Perceptron
# -------------------------------------------------

class MLP(nn.Module):

    def __init__(
        self,
        input_dim=28 * 28,
        hidden_dim=512,
        num_classes=10
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x


# -------------------------------------------------
# Convolutional Neural Network
# -------------------------------------------------

class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Conv Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


# -------------------------------------------------
# Quick test
# -------------------------------------------------

if __name__ == "__main__":

    sample = torch.randn(32, 1, 28, 28)

    mlp = MLP()
    cnn = CNN()

    print("MLP output:", mlp(sample).shape)
    print("CNN output:", cnn(sample).shape)