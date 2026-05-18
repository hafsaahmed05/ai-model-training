# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from dataset import get_dataloaders
from model import MLP, CNN


# -----------------------------
# Configuration
# -----------------------------

MODEL_TYPE = "cnn"      # "mlp" or "cnn"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "mlp"],
        default="cnn"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    return parser.parse_args()

# -----------------------------
# Training Function
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# -----------------------------
# Validation Function
# -----------------------------

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# -----------------------------
# Main Training Script
# -----------------------------

def main():
    args = parse_args()

    model_type = args.model
    epochs = args.epochs

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Load datasets
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Select model
    if model_type == "mlp":
        model = MLP()

    elif model_type == "cnn":
        model = CNN()

    else:
        raise ValueError("Invalid model_type")

    model = model.to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # History tracking
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    print(f"\nTraining on {DEVICE}")
    print(f"Model: {model_type.upper()}\n")

    # Epoch loop
    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print progress
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    # Save model checkpoint
    model_path = f"outputs/{model_type}_fashionmnist.pth"

    torch.save(model.state_dict(), model_path)

    print(f"\nModel saved to: {model_path}")

    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": [x / 100 for x in train_accuracies],
        "val_accuracy": [x / 100 for x in val_accuracies]
    }

    history_path = "outputs/training_log.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()