# dataset.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64, validation_split=0.1):
    """
    Loads Fashion-MNIST dataset and returns train, validation, and test DataLoaders.
    """

    # Transform:
    # 1. Convert image to tensor
    # 2. Normalize pixel values to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download training dataset
    full_train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Download test dataset
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Split training into train + validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    # Test one batch
    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Pixel value range:")
    print("Min:", images.min().item())
    print("Max:", images.max().item())