"""
gui_app.py  –  Part 1 Extra Credit: Training Control GUI
Tkinter window with:
  - Model selector (MLP / CNN)
  - Learning rate, batch size, and epoch count inputs
  - Start / Stop / Step-one-epoch buttons
  - Live matplotlib chart embedded in the window (loss + accuracy)

Requirements
------------
    pip install torch torchvision matplotlib

Run
---
    python gui_app.py

Notes
-----
Training runs in a background daemon thread so the GUI stays responsive.
The Stop button signals the thread to finish the current epoch and halt.
Step mode runs exactly one epoch then pauses (great for interactive exploration).
"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")                         # must be set before pyplot import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import get_dataloaders
from model import MLP, CNN


# ── Shared colour palette (mirrors visualize.py) ────────────────────────────────
TRAIN_COLOR = "#5271C4"
VAL_COLOR   = "#E25C3B"
TARGET_COLOR = "#2CA02C"


# ─────────────────────────────────────────────────────────────────────────────────
# Training worker – runs in a background thread
# ─────────────────────────────────────────────────────────────────────────────────
class TrainWorker:
    """
    Encapsulates the training loop.  Results are pushed onto a thread-safe
    queue so the GUI can poll and update the live chart without blocking.

    Queue message format:
        {"type": "epoch", "epoch": int, "train_loss": float, "val_loss": float,
         "train_acc": float, "val_acc": float}
        {"type": "done",  "reason": str}
        {"type": "error", "message": str}
    """

    def __init__(self, model_name, lr, batch_size, total_epochs,
                 result_queue, stop_event, step_event=None):
        self.model_name    = model_name
        self.lr            = lr
        self.batch_size    = batch_size
        self.total_epochs  = total_epochs
        self.result_queue  = result_queue
        self.stop_event    = stop_event   # threading.Event – set to request stop
        self.step_event    = step_event   # threading.Event – if set, run 1 epoch then pause

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── single epoch ────────────────────────────────────────────────────────────
    def _run_epoch(self, model, loader, criterion, optimizer, train=True):
        model.train() if train else model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.no_grad() if not train else torch.enable_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += labels.size(0)

        return total_loss / total, correct / total

    # ── main loop ───────────────────────────────────────────────────────────────
    def run(self):
        try:
            train_loader, val_loader, _ = get_dataloaders(batch_size=self.batch_size)

            # Build model
            if self.model_name == "CNN":
                model = CNN(num_classes=10).to(self.device)
            else:
                model = MLP(input_dim=28 * 28, hidden_dim=512,
                            num_classes=10).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=self.lr)

            for epoch in range(1, self.total_epochs + 1):
                # ── Check for stop request ───────────────────────────────────
                if self.stop_event.is_set():
                    self.result_queue.put({"type": "done", "reason": "stopped"})
                    return

                # ── Run one epoch ────────────────────────────────────────────
                train_loss, train_acc = self._run_epoch(
                    model, train_loader, criterion, optimizer, train=True)
                val_loss,   val_acc   = self._run_epoch(
                    model, val_loader,   criterion, optimizer, train=False)

                self.result_queue.put({
                    "type":       "epoch",
                    "epoch":      epoch,
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                })

                # ── Step mode: pause after each epoch ───────────────────────
                if self.step_event is not None:
                    self.step_event.clear()
                    # Block until step_event is set again (next Step press)
                    # or until stop is requested
                    while not self.step_event.is_set() and not self.stop_event.is_set():
                        threading.Event().wait(timeout=0.05)   # lightweight poll
                    if self.stop_event.is_set():
                        self.result_queue.put({"type": "done", "reason": "stopped"})
                        return

            # Save weights and log after training completes
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/model_weights.pth")

            self.result_queue.put({"type": "done", "reason": "completed"})

        except Exception as exc:
            self.result_queue.put({"type": "error", "message": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────────
# Main GUI Application
# ─────────────────────────────────────────────────────────────────────────────────
class TrainingGUI:
    # ── Poll interval for the result queue (ms) ──────────────────────────────────
    POLL_MS = 200

    def __init__(self, root):
        self.root = root
        self.root.title("CS 461 – Fashion-MNIST Training Control")
        self.root.resizable(True, True)

        # State
        self.train_thread  = None
        self.result_queue  = queue.Queue()
        self.stop_event    = threading.Event()
        self.step_event    = threading.Event()
        self.step_mode     = False   # toggled by "Step" button

        # Logged history (for the live chart)
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
        }

        self._build_control_panel()
        self._build_chart_panel()
        self._build_status_bar()

    # ─────────────────────────────────────────────────────────────────────────────
    # Layout builders
    # ─────────────────────────────────────────────────────────────────────────────
    def _build_control_panel(self):
        """Left-hand panel: all controls."""
        frame = ttk.LabelFrame(self.root, text="Training configuration", padding=12)
        frame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.root.columnconfigure(0, weight=0)

        row = 0

        # ── Model type ───────────────────────────────────────────────────────────
        ttk.Label(frame, text="Model type").grid(row=row, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar(value="CNN")
        model_menu = ttk.Combobox(frame, textvariable=self.model_var,
                                  values=["CNN", "MLP"], state="readonly", width=12)
        model_menu.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        row += 1

        # ── Learning rate ────────────────────────────────────────────────────────
        ttk.Label(frame, text="Learning rate").grid(row=row, column=0, sticky="w", pady=4)
        self.lr_var = tk.StringVar(value="0.001")
        lr_entry = ttk.Combobox(frame, textvariable=self.lr_var,
                                values=["0.1", "0.01", "0.001", "0.0001"],
                                width=12)
        lr_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        row += 1

        # ── Batch size ───────────────────────────────────────────────────────────
        ttk.Label(frame, text="Batch size").grid(row=row, column=0, sticky="w", pady=4)
        self.batch_var = tk.StringVar(value="64")
        batch_menu = ttk.Combobox(frame, textvariable=self.batch_var,
                                  values=["32", "64", "128", "256", "512"],
                                  state="readonly", width=12)
        batch_menu.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        row += 1

        # ── Epochs ───────────────────────────────────────────────────────────────
        ttk.Label(frame, text="Epochs").grid(row=row, column=0, sticky="w", pady=4)
        self.epoch_var = tk.IntVar(value=10)
        epoch_spin = ttk.Spinbox(frame, from_=1, to=100,
                                  textvariable=self.epoch_var, width=12)
        epoch_spin.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        row += 1

        # ── Separator ────────────────────────────────────────────────────────────
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # ── Start button ─────────────────────────────────────────────────────────
        self.start_btn = ttk.Button(frame, text="▶  Start training",
                                    command=self._on_start)
        self.start_btn.grid(row=row, column=0, columnspan=2,
                            sticky="ew", pady=(0, 4))
        row += 1

        # ── Stop button ──────────────────────────────────────────────────────────
        self.stop_btn = ttk.Button(frame, text="■  Stop",
                                   command=self._on_stop, state="disabled")
        self.stop_btn.grid(row=row, column=0, columnspan=2,
                           sticky="ew", pady=(0, 4))
        row += 1

        # ── Step button ──────────────────────────────────────────────────────────
        self.step_btn = ttk.Button(frame, text="⏭  Step one epoch",
                                   command=self._on_step, state="disabled")
        self.step_btn.grid(row=row, column=0, columnspan=2,
                           sticky="ew", pady=(0, 4))
        row += 1

        # ── Step-mode checkbox ───────────────────────────────────────────────────
        self.step_mode_var = tk.BooleanVar(value=False)
        step_check = ttk.Checkbutton(frame, text="Step mode (pause each epoch)",
                                     variable=self.step_mode_var)
        step_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1

        # ── Reset / clear chart ──────────────────────────────────────────────────
        reset_btn = ttk.Button(frame, text="↺  Reset chart", command=self._on_reset)
        reset_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        # Allow the Combobox column to expand
        frame.columnconfigure(1, weight=1)

    def _build_chart_panel(self):
        """Right-hand panel: embedded matplotlib figure."""
        frame = ttk.LabelFrame(self.root, text="Live training metrics", padding=8)
        frame.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(8, 4.5), tight_layout=True)
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_acc  = self.fig.add_subplot(1, 2, 2)
        self._init_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _build_status_bar(self):
        """Bottom status bar showing current epoch / accuracy."""
        self.status_var = tk.StringVar(value="Ready.")
        bar = ttk.Label(self.root, textvariable=self.status_var,
                        relief="sunken", anchor="w", padding=(6, 2))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 8))

    # ─────────────────────────────────────────────────────────────────────────────
    # Axes helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def _init_axes(self):
        """Set up empty axes with labels and style."""
        for ax, title, ylabel in [
            (self.ax_loss, "Loss", "Cross-entropy loss"),
            (self.ax_acc,  "Accuracy", "Accuracy (%)"),
        ]:
            ax.cla()
            ax.set_title(title, fontsize=11, fontweight="semibold")
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Placeholder text so the chart doesn't look blank before training starts
        for ax in (self.ax_loss, self.ax_acc):
            ax.text(0.5, 0.5, "Waiting for training…",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color="gray", fontsize=9)

    def _redraw_chart(self):
        """Redraw both subplots from the current history."""
        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        if not epochs:
            return

        # ── Loss ──────────────────────────────────────────────────────────────────
        self.ax_loss.cla()
        self.ax_loss.plot(epochs, self.history["train_loss"],
                          color=TRAIN_COLOR, linewidth=2, label="Train")
        self.ax_loss.plot(epochs, self.history["val_loss"],
                          color=VAL_COLOR, linewidth=2,
                          marker="o", markersize=3, label="Val")
        self.ax_loss.set_title("Loss", fontsize=11, fontweight="semibold")
        self.ax_loss.set_xlabel("Epoch", fontsize=9)
        self.ax_loss.set_ylabel("Cross-entropy loss", fontsize=9)
        self.ax_loss.legend(fontsize=8)
        self.ax_loss.grid(True, alpha=0.25, linestyle="--")
        self.ax_loss.spines["top"].set_visible(False)
        self.ax_loss.spines["right"].set_visible(False)

        # ── Accuracy ──────────────────────────────────────────────────────────────
        train_pct = [a * 100 for a in self.history["train_acc"]]
        val_pct   = [a * 100 for a in self.history["val_acc"]]

        self.ax_acc.cla()
        self.ax_acc.plot(epochs, train_pct,
                         color=TRAIN_COLOR, linewidth=2, label="Train")
        self.ax_acc.plot(epochs, val_pct,
                         color=VAL_COLOR, linewidth=2,
                         marker="o", markersize=3, label="Val")
        self.ax_acc.axhline(91, color=TARGET_COLOR, linewidth=1.2,
                            linestyle="--", alpha=0.7, label="91% target")
        self.ax_acc.set_ylim(max(0, min(train_pct + val_pct) - 5), 100)
        self.ax_acc.set_title("Accuracy", fontsize=11, fontweight="semibold")
        self.ax_acc.set_xlabel("Epoch", fontsize=9)
        self.ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
        self.ax_acc.legend(fontsize=8)
        self.ax_acc.grid(True, alpha=0.25, linestyle="--")
        self.ax_acc.spines["top"].set_visible(False)
        self.ax_acc.spines["right"].set_visible(False)

        self.fig.tight_layout()
        self.canvas.draw_idle()   # non-blocking redraw

    # ─────────────────────────────────────────────────────────────────────────────
    # Button callbacks
    # ─────────────────────────────────────────────────────────────────────────────
    def _on_start(self):
        """Validate inputs and launch training in a background thread."""
        # ── Validate ──────────────────────────────────────────────────────────────
        try:
            lr         = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            epochs     = int(self.epoch_var.get())
            assert lr > 0 and batch_size > 0 and epochs > 0
        except (ValueError, AssertionError):
            messagebox.showerror("Invalid input",
                                 "Please enter valid numeric values for all fields.")
            return

        model_name = self.model_var.get()
        self.step_mode = self.step_mode_var.get()

        # ── Reset state ───────────────────────────────────────────────────────────
        self.stop_event.clear()
        self.step_event.clear()
        if self.step_mode:
            self.step_event.set()      # allow first epoch to run immediately
        self.history = {k: [] for k in self.history}
        self._init_axes()
        self.canvas.draw()

        # ── Build worker ──────────────────────────────────────────────────────────
        worker = TrainWorker(
            model_name    = model_name,
            lr            = lr,
            batch_size    = batch_size,
            total_epochs  = epochs,
            result_queue  = self.result_queue,
            stop_event    = self.stop_event,
            step_event    = self.step_event if self.step_mode else None,
        )

        self.train_thread = threading.Thread(target=worker.run, daemon=True)
        self.train_thread.start()

        # ── Update button states ──────────────────────────────────────────────────
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.step_btn.config(state="normal" if self.step_mode else "disabled")
        self.status_var.set(f"Training {model_name} | lr={lr} | batch={batch_size} | epochs={epochs}")

        # ── Start polling ─────────────────────────────────────────────────────────
        self.root.after(self.POLL_MS, self._poll_queue)

    def _on_stop(self):
        """Signal the worker thread to stop after the current epoch."""
        self.stop_event.set()
        # Also unblock a paused step-mode thread
        self.step_event.set()
        self.status_var.set("Stop requested – finishing current epoch…")

    def _on_step(self):
        """Advance exactly one epoch in step mode."""
        self.step_event.set()
        self.status_var.set("Step → running one epoch…")

    def _on_reset(self):
        """Clear the chart and history (only available when not training)."""
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showinfo("Training in progress",
                                "Stop training before resetting.")
            return
        self.history = {k: [] for k in self.history}
        self._init_axes()
        self.canvas.draw()
        self.status_var.set("Chart cleared. Ready.")

    # ─────────────────────────────────────────────────────────────────────────────
    # Queue polling
    # ─────────────────────────────────────────────────────────────────────────────
    def _poll_queue(self):
        """
        Called periodically by the Tk event loop.
        Drains all pending messages and updates the chart.
        """
        updated = False
        try:
            while True:
                msg = self.result_queue.get_nowait()
                self._handle_message(msg)
                updated = True
        except queue.Empty:
            pass

        if updated:
            self._redraw_chart()

        # Keep polling while the thread is alive
        if self.train_thread and self.train_thread.is_alive():
            self.root.after(self.POLL_MS, self._poll_queue)

    def _handle_message(self, msg):
        """Process a single message from the training worker."""
        if msg["type"] == "epoch":
            ep = msg["epoch"]
            self.history["train_loss"].append(msg["train_loss"])
            self.history["val_loss"].append(msg["val_loss"])
            self.history["train_acc"].append(msg["train_acc"])
            self.history["val_acc"].append(msg["val_acc"])

            self.status_var.set(
                f"Epoch {ep} complete  │  "
                f"train loss: {msg['train_loss']:.4f}  val loss: {msg['val_loss']:.4f}  │  "
                f"val acc: {msg['val_acc']*100:.2f}%"
            )

        elif msg["type"] == "done":
            reason = msg["reason"]
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.step_btn.config(state="disabled")
            if reason == "completed":
                self.status_var.set(
                    f"Training complete! "
                    f"Final val acc: {self.history['val_acc'][-1]*100:.2f}%  "
                    f"– weights saved to outputs/model_weights.pth"
                )
                self._save_log()
            else:
                self.status_var.set("Training stopped by user.")

        elif msg["type"] == "error":
            messagebox.showerror("Training error", msg["message"])
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.step_btn.config(state="disabled")
            self.status_var.set(f"Error: {msg['message']}")

    # ─────────────────────────────────────────────────────────────────────────────
    # Save log (mirrors format expected by visualize.py)
    # ─────────────────────────────────────────────────────────────────────────────
    def _save_log(self):
        os.makedirs("outputs", exist_ok=True)
        log_path = "outputs/training_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "train_loss":     self.history["train_loss"],
                "val_loss":       self.history["val_loss"],
                "train_accuracy": self.history["train_acc"],
                "val_accuracy":   self.history["val_acc"],
            }, f, indent=2)
        print(f"[gui] Training log saved → {log_path}")


# ─────────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()