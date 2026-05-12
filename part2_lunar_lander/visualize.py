"""
visualize.py  –  Part 2: Lunar Lander DQN
Reads training reward logs and produces all plots required by the rubric.

Expected inputs
---------------
  outputs/training_rewards.json     – written by train_dqn.py (see note below)
  outputs/mod_rewards.json          – same format, from your modification run
  outputs/eval_rewards.json         – written by evaluate.py

Outputs saved to outputs/
--------------------------------------
  reward_curve.png          episodic reward + 100-ep moving average
  baseline_vs_mod.png       side-by-side comparison of baseline and modification
  eval_distribution.png     histogram of evaluation episode rewards
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Shared style ────────────────────────────────────────────────────────────────
BASELINE_COLOR  = "#5271C4"   # blue
MOD_COLOR       = "#E25C3B"   # coral
AVG_COLOR       = "#F5A623"   # amber  – moving average line
EVAL_COLOR      = "#2CA02C"   # green
GRID_ALPHA      = 0.25
MOVING_WINDOW   = 100         # episodes in the moving average


# ── Helpers ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Plot DQN training curves.")
    parser.add_argument("--baseline", type=str, default="outputs/training_rewards.json",
                        help="JSON file with baseline training rewards.")
    parser.add_argument("--mod", type=str, default=None,
                        help="JSON file with modification training rewards (optional).")
    parser.add_argument("--eval", type=str, default="outputs/eval_rewards.json",
                        help="JSON file with evaluation rewards from evaluate.py.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def load_rewards(path):
    """Load a rewards JSON file; returns list of floats."""
    with open(path, "r") as f:
        data = json.load(f)
    # Support both {"rewards": [...]} and a bare list
    if isinstance(data, dict):
        return data["rewards"]
    return data


def moving_average(rewards, window):
    """Compute a trailing moving average; first (window-1) values use expanding window."""
    result = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        result.append(np.mean(rewards[start : i + 1]))
    return result


def style_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(fontsize=9)


# ── Plot 1: reward curve + moving average ───────────────────────────────────────
def plot_reward_curve(rewards, output_dir, label="Baseline", color=BASELINE_COLOR,
                      filename="reward_curve.png"):
    """
    Single training run: raw episodic reward (faint) overlaid with the
    100-episode moving average (bold).  Includes a horizontal reference
    line at the 100-point minimum threshold and 200-point solved threshold.
    """
    episodes = list(range(1, len(rewards) + 1))
    ma       = moving_average(rewards, MOVING_WINDOW)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Raw rewards – very transparent so they don't drown the MA
    ax.plot(episodes, rewards, color=color, alpha=0.18, linewidth=0.8,
            label=f"{label} (raw)")

    # 100-ep moving average
    ax.plot(episodes, ma, color=color, linewidth=2.2,
            label=f"{label} ({MOVING_WINDOW}-ep MA)")

    # Threshold reference lines
    ax.axhline(200, color="#2CA02C", linewidth=1.2, linestyle="--",
               alpha=0.7, label="Solved threshold (200)")
    ax.axhline(100, color="#9467BD", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Minimum threshold (100)")

    # Annotate where MA first crosses 100
    cross_idx = next((i for i, v in enumerate(ma) if v >= 100), None)
    if cross_idx is not None:
        ax.annotate(
            f"MA ≥ 100\n(ep {cross_idx + 1})",
            xy=(cross_idx + 1, ma[cross_idx]),
            xytext=(cross_idx + 1 + max(len(rewards) * 0.03, 5),
                    ma[cross_idx] + 20),
            fontsize=8, color="#9467BD",
            arrowprops=dict(arrowstyle="->", color="#9467BD", lw=1),
        )

    style_ax(ax,
             title=f"DQN Training – Episodic Reward ({label})",
             xlabel="Episode",
             ylabel="Total reward")

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Reward curve saved → {path}")


# ── Plot 2: baseline vs modification comparison ─────────────────────────────────
def plot_baseline_vs_mod(baseline_rewards, mod_rewards, output_dir):
    """
    Two-panel figure: left = raw + MA for both runs overlaid on one axis,
    right = bar chart of final 100-episode average reward.
    """
    ep_base = list(range(1, len(baseline_rewards) + 1))
    ep_mod  = list(range(1, len(mod_rewards) + 1))
    ma_base = moving_average(baseline_rewards, MOVING_WINDOW)
    ma_mod  = moving_average(mod_rewards,      MOVING_WINDOW)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.3)

    ax_curve = fig.add_subplot(gs[0])
    ax_bar   = fig.add_subplot(gs[1])

    # ── Learning curves ──────────────────────────────────────────────────────────
    ax_curve.plot(ep_base, baseline_rewards, color=BASELINE_COLOR,
                  alpha=0.15, linewidth=0.7)
    ax_curve.plot(ep_base, ma_base, color=BASELINE_COLOR, linewidth=2.2,
                  label=f"Baseline ({MOVING_WINDOW}-ep MA)")

    ax_curve.plot(ep_mod, mod_rewards, color=MOD_COLOR,
                  alpha=0.15, linewidth=0.7)
    ax_curve.plot(ep_mod, ma_mod, color=MOD_COLOR, linewidth=2.2,
                  label=f"Modified ({MOVING_WINDOW}-ep MA)")

    ax_curve.axhline(200, color="#2CA02C", linewidth=1.1, linestyle="--",
                     alpha=0.6, label="Solved (200)")
    ax_curve.axhline(100, color="#9467BD", linewidth=1.1, linestyle=":",
                     alpha=0.6, label="Minimum (100)")

    style_ax(ax_curve,
             title="Baseline vs Modification – Learning Curves",
             xlabel="Episode",
             ylabel="Total reward")

    # ── Bar chart: final-100-ep average ─────────────────────────────────────────
    final_base = float(np.mean(baseline_rewards[-100:]))
    final_mod  = float(np.mean(mod_rewards[-100:]))

    bars = ax_bar.bar(
        ["Baseline", "Modified"],
        [final_base, final_mod],
        color=[BASELINE_COLOR, MOD_COLOR],
        width=0.5,
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels on bars
    for bar, val in zip(bars, [final_base, final_mod]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax_bar.axhline(200, color="#2CA02C", linewidth=1.1, linestyle="--", alpha=0.6)
    ax_bar.axhline(100, color="#9467BD", linewidth=1.1, linestyle=":",  alpha=0.6)
    ax_bar.set_ylim(min(0, min(final_base, final_mod) - 30),
                    max(250, max(final_base, final_mod) + 40))
    style_ax(ax_bar,
             title="Final 100-ep\nAverage Reward",
             xlabel="",
             ylabel="Avg reward",
             legend=False)

    fig.suptitle("Baseline vs Modification Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "baseline_vs_mod.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Comparison plot saved → {path}")


# ── Plot 3: evaluation reward distribution ──────────────────────────────────────
def plot_eval_distribution(eval_rewards, output_dir):
    """
    Histogram of per-episode rewards from the 100-episode evaluation run,
    with vertical lines for mean and the two rubric thresholds.
    """
    avg = float(np.mean(eval_rewards))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(eval_rewards, bins=20, color=EVAL_COLOR, alpha=0.75,
            edgecolor="white", linewidth=0.6)

    ax.axvline(avg,  color="black",   linewidth=2,   linestyle="-",
               label=f"Mean: {avg:.1f}")
    ax.axvline(200,  color="#2CA02C", linewidth=1.5,  linestyle="--",
               label="Solved (200)")
    ax.axvline(100,  color="#9467BD", linewidth=1.5,  linestyle=":",
               label="Minimum (100)")

    style_ax(ax,
             title="Evaluation Episode Reward Distribution (ε = 0)",
             xlabel="Total reward",
             ylabel="Count")

    fig.tight_layout()
    path = os.path.join(output_dir, "eval_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Eval distribution saved → {path}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Baseline rewards ─────────────────────────────────────────────────────────
    if not os.path.exists(args.baseline):
        print(f"[visualize] Baseline log not found at {args.baseline}. "
              "Add the JSON-save snippet to train_dqn.py and re-run training.")
        return

    baseline_rewards = load_rewards(args.baseline)
    print(f"[visualize] Loaded {len(baseline_rewards)} baseline episodes.")

    plot_reward_curve(baseline_rewards, args.output_dir,
                      label="Baseline", color=BASELINE_COLOR,
                      filename="reward_curve.png")

    # ── Modification comparison (optional) ───────────────────────────────────────
    if args.mod:
        if not os.path.exists(args.mod):
            print(f"[visualize] Mod log not found at {args.mod}. Skipping comparison.")
        else:
            mod_rewards = load_rewards(args.mod)
            print(f"[visualize] Loaded {len(mod_rewards)} modification episodes.")
            # Also draw the mod curve on its own for the report appendix
            plot_reward_curve(mod_rewards, args.output_dir,
                              label="Modified", color=MOD_COLOR,
                              filename="mod_reward_curve.png")
            plot_baseline_vs_mod(baseline_rewards, mod_rewards, args.output_dir)
    else:
        print("[visualize] No --mod file provided; skipping baseline vs mod plot.")

    # ── Evaluation distribution ───────────────────────────────────────────────────
    if os.path.exists(args.eval):
        eval_data    = json.load(open(args.eval))
        eval_rewards = eval_data["rewards"]
        print(f"[visualize] Loaded {len(eval_rewards)} evaluation episodes.")
        plot_eval_distribution(eval_rewards, args.output_dir)
    else:
        print(f"[visualize] Eval log not found at {args.eval}. "
              "Run evaluate.py first.")


if __name__ == "__main__":
    main()