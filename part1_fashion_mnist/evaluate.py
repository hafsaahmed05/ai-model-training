"""
evaluate.py  –  Part 1: Fashion-MNIST
Loads a trained model checkpoint and runs evaluation on the held-out test set.

Outputs
-------
- Prints overall test accuracy
- Saves outputs/confusion_matrix.png
- Saves outputs/misclassified_samples.png  (first N misclassifications)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from model import MLP, CNN


# ── Human-readable class names (FashionMNIST label order) ──────────────────────
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]


# ── Argument parsing ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Fashion-MNIST model.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved model weights (.pth).")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="cnn",
                        help="Architecture to load weights into.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_misclassified", type=int, default=16,
                        help="How many misclassified examples to display.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


# ── Core evaluation loop ────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """
    Run model inference over the full loader.

    Returns
    -------
    all_preds   : np.ndarray  – predicted class indices
    all_labels  : np.ndarray  – ground-truth class indices
    all_images  : np.ndarray  – raw image tensors (N, 1, 28, 28)
    accuracy    : float       – top-1 accuracy in [0, 1]
    """
    model.eval()
    all_preds, all_labels, all_images = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                     # (B, 10) logits
            preds = outputs.argmax(dim=1)               # predicted class

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_images.append(images.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_images = np.concatenate(all_images)

    accuracy = (all_preds == all_labels).mean()
    return all_preds, all_labels, all_images, accuracy


# ── Confusion matrix plot ───────────────────────────────────────────────────────
def plot_confusion_matrix(all_labels, all_preds, output_dir):
    """Generate and save a normalised confusion matrix heat-map."""
    cm = confusion_matrix(all_labels, all_preds, normalize="true")  # row-normalised

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format=".2f",   # show proportion, not raw count
        xticks_rotation=45,
    )
    ax.set_title("Normalised confusion matrix – Fashion-MNIST test set", pad=14)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ── Misclassified samples grid ──────────────────────────────────────────────────
def plot_misclassified(all_images, all_labels, all_preds, n, output_dir):
    """
    Display a grid of the first `n` misclassified test images.
    Title of each cell: 'True: X  Pred: Y'
    """
    # Indices where prediction differs from ground truth
    wrong_idx = np.where(all_preds != all_labels)[0]

    if len(wrong_idx) == 0:
        print("[evaluate] No misclassified samples – perfect test accuracy!")
        return

    n = min(n, len(wrong_idx))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    for i, idx in enumerate(wrong_idx[:n]):
        img = all_images[idx, 0]            # squeeze channel dim  (28, 28)
        true_label = CLASS_NAMES[all_labels[idx]]
        pred_label = CLASS_NAMES[all_preds[idx]]

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            fontsize=8,
            color="red",
        )
        axes[i].axis("off")

    # Hide any unused subplot slots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"First {n} misclassified samples", fontsize=12, y=1.01)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "misclassified_samples.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Misclassified grid saved → {save_path}")


# ── Per-class accuracy breakdown ────────────────────────────────────────────────
def print_per_class_accuracy(all_labels, all_preds):
    """Print a tidy table of per-class accuracy scores."""
    print("\n── Per-class accuracy ───────────────────────────────")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        bar = "█" * int(cls_acc // 5)   # simple ASCII progress bar
        print(f"  {cls_name:<15s}  {cls_acc:5.1f}%  {bar}")
    print("─────────────────────────────────────────────────────\n")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"outputs/{args.model}_fashionmnist.pth"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device: {device}")

    # Load data (we only need the test split here)
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Instantiate the correct architecture
    if args.model == "cnn":
        model = CNN(num_classes=10).to(device)
    else:
        model = MLP(input_dim=28 * 28, hidden_dim=512, num_classes=10).to(device)

    # Load saved weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Support both raw state-dict saves and wrapped {"model_state": ...} saves
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    print(f"[evaluate] Loaded weights from {args.checkpoint}")

    # Run evaluation
    all_preds, all_labels, all_images, accuracy = evaluate(model, test_loader, device)

    print(f"\n[evaluate] Test accuracy: {accuracy * 100:.2f}%")
    if accuracy >= 0.91:
        print("[evaluate] ✓ Target of ≥91% reached!")
    else:
        print("[evaluate] ✗ Below 91% target – see report analysis section.")

    print_per_class_accuracy(all_labels, all_preds)

    # Generate and save plots
    plot_confusion_matrix(all_labels, all_preds, args.output_dir)
    plot_misclassified(all_images, all_labels, all_preds,
                       args.num_misclassified, args.output_dir)


if __name__ == "__main__":
    main()