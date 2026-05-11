"""
visualize.py  –  Part 1: Fashion-MNIST
Plots training and validation curves logged during training.

Expected input: a JSON log file written by train.py with the schema:
    {
      "train_loss":     [float, ...],   # one entry per epoch
      "val_loss":       [float, ...],
      "train_accuracy": [float, ...],   # values in [0, 1]
      "val_accuracy":   [float, ...]
    }

Outputs saved to outputs/
-------------------------------
  loss_curve.png
  accuracy_curve.png
  combined_dashboard.png   (both plots side-by-side, good for the report)

Usage
-----
    python visualize.py --log outputs/training_log.json
    python visualize.py --log outputs/training_log.json --output_dir outputs
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Shared style ────────────────────────────────────────────────────────────────
TRAIN_COLOR = "#5271C4"   # blue  – solid line
VAL_COLOR   = "#E25C3B"   # coral – solid line with markers
GRID_ALPHA  = 0.25


def apply_style(ax, title, xlabel, ylabel, legend=True):
    """Apply consistent formatting to a single Axes object."""
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(fontsize=9)


# ── Parse args ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Plot Fashion-MNIST training curves.")
    parser.add_argument("--log", type=str, default="outputs/training_log.json",
                        help="Path to the JSON log file produced by train.py.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Where to save the output figures.")
    return parser.parse_args()


# ── Individual plots ────────────────────────────────────────────────────────────
def plot_loss(epochs, train_loss, val_loss, output_dir):
    """Plot training vs validation loss per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, train_loss, color=TRAIN_COLOR, linewidth=2, label="Train loss")
    ax.plot(epochs, val_loss,   color=VAL_COLOR,   linewidth=2,
            marker="o", markersize=4, label="Val loss")

    # Annotate minimum validation loss
    best_epoch = int(np.argmin(val_loss))
    best_val   = val_loss[best_epoch]
    ax.annotate(
        f"best val\n{best_val:.4f}",
        xy=(epochs[best_epoch], best_val),
        xytext=(epochs[best_epoch] + 0.5, best_val + (max(val_loss) - min(val_loss)) * 0.15),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        color="gray",
    )

    apply_style(ax,
                title="Training vs validation loss",
                xlabel="Epoch",
                ylabel="Cross-entropy loss")

    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Loss curve saved → {path}")


def plot_accuracy(epochs, train_acc, val_acc, output_dir):
    """Plot training vs validation accuracy per epoch."""
    # Convert to percentages for readability
    train_pct = [a * 100 for a in train_acc]
    val_pct   = [a * 100 for a in val_acc]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(epochs, train_pct, color=TRAIN_COLOR, linewidth=2, label="Train accuracy")
    ax.plot(epochs, val_pct,   color=VAL_COLOR,   linewidth=2,
            marker="o", markersize=4, label="Val accuracy")

    # Draw the 91% target line
    ax.axhline(91, color="#2CA02C", linewidth=1.2, linestyle="--", alpha=0.7, label="91% target")

    # Annotate best validation accuracy
    best_epoch = int(np.argmax(val_pct))
    best_val   = val_pct[best_epoch]
    ax.annotate(
        f"best val\n{best_val:.1f}%",
        xy=(epochs[best_epoch], best_val),
        xytext=(epochs[best_epoch] + 0.5, best_val - 4),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        color="gray",
    )

    ax.set_ylim(max(0, min(train_pct + val_pct) - 5), 100)
    apply_style(ax,
                title="Training vs validation accuracy",
                xlabel="Epoch",
                ylabel="Accuracy (%)")

    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Accuracy curve saved → {path}")


# ── Combined dashboard ──────────────────────────────────────────────────────────
def plot_combined_dashboard(epochs, train_loss, val_loss,
                             train_acc, val_acc, output_dir):
    """
    Single figure with loss on the left and accuracy on the right.
    Suitable for dropping directly into the report.
    """
    train_pct = [a * 100 for a in train_acc]
    val_pct   = [a * 100 for a in val_acc]

    fig = plt.figure(figsize=(13, 4.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0])
    ax_acc  = fig.add_subplot(gs[1])

    # ── Loss ──
    ax_loss.plot(epochs, train_loss, color=TRAIN_COLOR, linewidth=2, label="Train")
    ax_loss.plot(epochs, val_loss,   color=VAL_COLOR,   linewidth=2,
                 marker="o", markersize=4, label="Val")
    apply_style(ax_loss,
                title="Loss per epoch",
                xlabel="Epoch",
                ylabel="Cross-entropy loss")

    # ── Accuracy ──
    ax_acc.plot(epochs, train_pct, color=TRAIN_COLOR, linewidth=2, label="Train")
    ax_acc.plot(epochs, val_pct,   color=VAL_COLOR,   linewidth=2,
                marker="o", markersize=4, label="Val")
    ax_acc.axhline(91, color="#2CA02C", linewidth=1.2, linestyle="--",
                   alpha=0.7, label="91% target")
    ax_acc.set_ylim(max(0, min(train_pct + val_pct) - 5), 100)
    apply_style(ax_acc,
                title="Accuracy per epoch",
                xlabel="Epoch",
                ylabel="Accuracy (%)")

    fig.suptitle("Fashion-MNIST training summary", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "combined_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Combined dashboard saved → {path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load log
    with open(args.log, "r") as f:
        log = json.load(f)

    train_loss = log["train_loss"]
    val_loss   = log["val_loss"]
    train_acc  = log["train_accuracy"]
    val_acc    = log["val_accuracy"]

    n_epochs = len(train_loss)
    epochs   = list(range(1, n_epochs + 1))

    print(f"[visualize] Loaded log with {n_epochs} epochs.")
    print(f"  Final train loss:  {train_loss[-1]:.4f}  |  val loss:  {val_loss[-1]:.4f}")
    print(f"  Final train acc:   {train_acc[-1]*100:.2f}%  |  val acc:   {val_acc[-1]*100:.2f}%")

    plot_loss(epochs, train_loss, val_loss, args.output_dir)
    plot_accuracy(epochs, train_acc, val_acc, args.output_dir)
    plot_combined_dashboard(epochs, train_loss, val_loss,
                             train_acc, val_acc, args.output_dir)


if __name__ == "__main__":
    main()