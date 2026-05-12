"""
main.py  –  Part 2: Lunar Lander DQN
Unified entry point for the entire Part 2 pipeline.

Modes
-----
  train       Run DQN training and save model + reward log
  evaluate    Load a checkpoint and run 100 greedy evaluation episodes
  visualize   Generate all plots from saved reward logs
  demo        Run visual demo episodes (live window or recorded video)
  all         Run train → evaluate → visualize in sequence (no demo)

Usage examples
--------------
  # Full pipeline from scratch:
  python main.py all

  # Train only with custom hyperparameters:
  python main.py train --episodes 1000 --lr 5e-4 --batch 128

  # Evaluate a specific checkpoint:
  python main.py evaluate --checkpoint outputs/model_weights.pth

  # Visualize with a modification comparison:
  python main.py visualize --mod outputs/mod_rewards.json

  # Record a demo video:
  python main.py demo --mode record --episodes 3

  # Live window demo:
  python main.py demo --mode live
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from dqn_agent import DQNAgent
from env_wrappers import make_env
from replay_buffer import ReplayBuffer


# ══════════════════════════════════════════════════════════════════════════════════
# Default hyperparameters  (override via CLI flags)
# ══════════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    episodes          = 1000,
    max_steps         = 1000,
    batch_size        = 64,
    buffer_capacity   = 100_000,
    target_update_freq= 10,
    learning_rate     = 1e-3,
    gamma             = 0.99,
    epsilon           = 1.0,
    epsilon_min       = 0.01,
    epsilon_decay     = 0.995,
    eval_episodes     = 100,
    checkpoint        = "outputs/model_weights.pth",
    output_dir        = "outputs",
)


# ══════════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════════
def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CS 461 – Part 2: Lunar Lander DQN pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the DQN agent.")
    p_train.add_argument("--episodes",    type=int,   default=DEFAULTS["episodes"])
    p_train.add_argument("--max_steps",   type=int,   default=DEFAULTS["max_steps"])
    p_train.add_argument("--batch",       type=int,   default=DEFAULTS["batch_size"],
                         dest="batch_size")
    p_train.add_argument("--buffer",      type=int,   default=DEFAULTS["buffer_capacity"],
                         dest="buffer_capacity")
    p_train.add_argument("--target_freq", type=int,   default=DEFAULTS["target_update_freq"],
                         dest="target_update_freq")
    p_train.add_argument("--lr",          type=float, default=DEFAULTS["learning_rate"],
                         dest="learning_rate")
    p_train.add_argument("--gamma",       type=float, default=DEFAULTS["gamma"])
    p_train.add_argument("--epsilon",     type=float, default=DEFAULTS["epsilon"])
    p_train.add_argument("--epsilon_min", type=float, default=DEFAULTS["epsilon_min"])
    p_train.add_argument("--epsilon_decay",type=float,default=DEFAULTS["epsilon_decay"])
    p_train.add_argument("--output_dir",  type=str,   default=DEFAULTS["output_dir"])
    p_train.add_argument("--tag",         type=str,   default=None,
                         help="Optional label for this run (e.g. 'mod_noise'). "
                              "Saves rewards to outputs/<tag>_rewards.json.")

    # ── evaluate ───────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved checkpoint (ε = 0).")
    p_eval.add_argument("--checkpoint",  type=str, default=DEFAULTS["checkpoint"])
    p_eval.add_argument("--episodes",    type=int, default=DEFAULTS["eval_episodes"])
    p_eval.add_argument("--max_steps",   type=int, default=DEFAULTS["max_steps"])
    p_eval.add_argument("--output_dir",  type=str, default=DEFAULTS["output_dir"])
    p_eval.add_argument("--render",      action="store_true",
                        help="Open a live render window during evaluation.")

    # ── visualize ──────────────────────────────────────────────────────────────
    p_vis = sub.add_parser("visualize", help="Generate all training/eval plots.")
    p_vis.add_argument("--baseline",    type=str,
                       default="outputs/training_rewards.json")
    p_vis.add_argument("--mod",         type=str, default=None,
                       help="Path to modification run reward log for comparison.")
    p_vis.add_argument("--eval",        type=str,
                       default="outputs/eval_rewards.json")
    p_vis.add_argument("--output_dir",  type=str, default=DEFAULTS["output_dir"])

    # ── demo ───────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo", help="Run visual demo episodes.")
    p_demo.add_argument("--checkpoint", type=str, default=DEFAULTS["checkpoint"])
    p_demo.add_argument("--episodes",   type=int, default=3)
    p_demo.add_argument("--max_steps",  type=int, default=DEFAULTS["max_steps"])
    p_demo.add_argument("--mode",       type=str, choices=["live", "record"],
                        default="record")
    p_demo.add_argument("--format",     type=str, choices=["mp4", "gif"],
                        default="mp4")
    p_demo.add_argument("--fps",        type=int, default=30)
    p_demo.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"])

    # ── all ────────────────────────────────────────────────────────────────────
    p_all = sub.add_parser("all",
                           help="Run train → evaluate → visualize in sequence.")
    p_all.add_argument("--episodes",    type=int,   default=DEFAULTS["episodes"])
    p_all.add_argument("--max_steps",   type=int,   default=DEFAULTS["max_steps"])
    p_all.add_argument("--batch",       type=int,   default=DEFAULTS["batch_size"],
                       dest="batch_size")
    p_all.add_argument("--lr",          type=float, default=DEFAULTS["learning_rate"],
                       dest="learning_rate")
    p_all.add_argument("--output_dir",  type=str,   default=DEFAULTS["output_dir"])

    return parser


# ══════════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════════
def run_train(args):
    print("\n" + "═" * 55)
    print("  TRAINING")
    print("═" * 55)
    print(f"  Episodes     : {args.episodes}")
    print(f"  Max steps    : {args.max_steps}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Buffer cap   : {args.buffer_capacity:,}")
    print(f"  Target freq  : every {args.target_update_freq} episodes")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Gamma        : {args.gamma}")
    print(f"  ε start→min  : {args.epsilon} → {args.epsilon_min}  "
          f"(decay {args.epsilon_decay})")
    print("═" * 55 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Environment ─────────────────────────────────────────────────────────────
    env = make_env()
    state_size  = env.observation_space.shape[0]   # 8
    action_size = env.action_space.n               # 4

    # ── Agent + buffer ──────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_size    = state_size,
        action_size   = action_size,
        learning_rate = args.learning_rate,
        gamma         = args.gamma,
        epsilon       = args.epsilon,
        epsilon_min   = args.epsilon_min,
        epsilon_decay = args.epsilon_decay,
    )

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    device = next(agent.q_network.parameters()).device
    print(f"[train] Device: {device}\n")

    # ── Episode loop ────────────────────────────────────────────────────────────
    episode_rewards = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        for _ in range(args.max_steps):
            action     = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            agent.update(replay_buffer, args.batch_size)

            state        = next_state
            total_reward += reward

            if done:
                break

        # Periodically sync target network
        if episode % args.target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)

        # Console logging – every episode, plus a moving-average summary
        ma_100 = float(np.mean(episode_rewards[-100:]))
        print(
            f"Ep {episode:>5d}/{args.episodes}  |  "
            f"reward: {total_reward:>8.2f}  |  "
            f"100-ep avg: {ma_100:>8.2f}  |  "
            f"ε: {agent.epsilon:.4f}"
        )

    env.close()

    # ── Save weights ─────────────────────────────────────────────────────────────
    weights_path = os.path.join(args.output_dir, "model_weights.pth")
    torch.save(agent.q_network.state_dict(), weights_path)
    print(f"\n[train] Model saved → {weights_path}")

    # ── Save reward log ──────────────────────────────────────────────────────────
    tag       = args.tag if hasattr(args, "tag") and args.tag else "training"
    log_name  = f"{tag}_rewards.json"
    log_path  = os.path.join(args.output_dir, log_name)
    with open(log_path, "w") as f:
        json.dump({"rewards": episode_rewards, "tag": tag}, f, indent=2)
    print(f"[train] Reward log saved → {log_path}")

    final_avg = float(np.mean(episode_rewards[-100:]))
    print(f"\n[train] Final 100-ep average reward: {final_avg:.2f}")
    if final_avg >= 200:
        print("[train] ★ Stretch goal reached (≥200)!")
    elif final_avg >= 100:
        print("[train] ✓ Minimum threshold met (≥100).")
    else:
        print("[train] ✗ Below minimum – consider more episodes or tuning.")

    return episode_rewards, weights_path


# ══════════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════════
def run_evaluate(args):
    """Thin wrapper that delegates to evaluate.py logic inline."""
    print("\n" + "═" * 55)
    print("  EVALUATION")
    print("═" * 55)

    # Import here so evaluate.py can also be run standalone
    from evaluate import run_evaluation, summarise

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

    render_mode = "human" if hasattr(args, "render") and args.render else None
    env = make_env(render_mode=render_mode)

    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size)
    weights = torch.load(args.checkpoint, map_location=device)
    agent.q_network.load_state_dict(weights)
    agent.q_network.eval()

    print(f"[evaluate] Loaded {args.checkpoint}")
    print(f"[evaluate] Running {args.episodes} evaluation episodes (ε = 0)…\n")

    rewards = run_evaluation(agent, env, args.episodes, args.max_steps)
    env.close()

    avg, success = summarise(rewards, args.output_dir)
    return rewards


# ══════════════════════════════════════════════════════════════════════════════════
# Visualize
# ══════════════════════════════════════════════════════════════════════════════════
def run_visualize(args):
    print("\n" + "═" * 55)
    print("  VISUALIZE")
    print("═" * 55)

    # Delegate to visualize.py functions
    from visualize import (
        load_rewards,
        plot_reward_curve,
        plot_baseline_vs_mod,
        plot_eval_distribution,
        BASELINE_COLOR,
        MOD_COLOR,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.baseline):
        print(f"[visualize] Baseline log not found: {args.baseline}")
        print("  Run  python main.py train  first.")
        return

    baseline_rewards = load_rewards(args.baseline)
    print(f"[visualize] Loaded {len(baseline_rewards)} baseline episodes.")
    plot_reward_curve(baseline_rewards, args.output_dir,
                      label="Baseline", color=BASELINE_COLOR,
                      filename="reward_curve.png")

    if args.mod:
        if not os.path.exists(args.mod):
            print(f"[visualize] Mod log not found: {args.mod}. Skipping comparison.")
        else:
            mod_rewards = load_rewards(args.mod)
            print(f"[visualize] Loaded {len(mod_rewards)} modification episodes.")
            plot_reward_curve(mod_rewards, args.output_dir,
                              label="Modified", color=MOD_COLOR,
                              filename="mod_reward_curve.png")
            plot_baseline_vs_mod(baseline_rewards, mod_rewards, args.output_dir)
    else:
        print("[visualize] No --mod provided. Skipping comparison plot.")

    if os.path.exists(args.eval):
        eval_data = json.load(open(args.eval))
        print(f"[visualize] Loaded {len(eval_data['rewards'])} eval episodes.")
        plot_eval_distribution(eval_data["rewards"], args.output_dir)
    else:
        print(f"[visualize] Eval log not found: {args.eval}. "
              "Run  python main.py evaluate  first.")


# ══════════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════════
def run_demo(args):
    print("\n" + "═" * 55)
    print("  DEMO")
    print("═" * 55)

    from render_demo import load_agent, run_live, run_record

    tmp_env     = make_env()
    state_size  = tmp_env.observation_space.shape[0]
    action_size = tmp_env.action_space.n
    tmp_env.close()

    agent = load_agent(args.checkpoint, state_size, action_size)

    if args.mode == "live":
        run_live(agent, args)
    else:
        run_record(agent, args)


# ══════════════════════════════════════════════════════════════════════════════════
# All  (train → evaluate → visualize)
# ══════════════════════════════════════════════════════════════════════════════════
def run_all(args):
    """
    Convenience pipeline: train, then immediately evaluate and plot.
    Uses DEFAULTS for anything not overridden on the CLI.
    """

    # ── Patch in defaults for sub-steps not exposed at the 'all' level ──────────
    args.buffer_capacity    = DEFAULTS["buffer_capacity"]
    args.target_update_freq = DEFAULTS["target_update_freq"]
    args.gamma              = DEFAULTS["gamma"]
    args.epsilon            = DEFAULTS["epsilon"]
    args.epsilon_min        = DEFAULTS["epsilon_min"]
    args.epsilon_decay      = DEFAULTS["epsilon_decay"]
    args.tag                = None

    # ── Train ────────────────────────────────────────────────────────────────────
    _, weights_path = run_train(args)

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    args.checkpoint = weights_path
    args.episodes   = DEFAULTS["eval_episodes"]
    args.render     = False
    run_evaluate(args)

    # ── Visualize ────────────────────────────────────────────────────────────────
    args.baseline = os.path.join(args.output_dir, "training_rewards.json")
    args.mod      = None
    args.eval     = os.path.join(args.output_dir, "eval_rewards.json")
    run_visualize(args)

    print("\n" + "═" * 55)
    print("  Pipeline complete.  Check outputs/ for all files.")
    print("═" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":      run_train,
        "evaluate":   run_evaluate,
        "visualize":  run_visualize,
        "demo":       run_demo,
        "all":        run_all,
    }

    dispatch[args.mode](args)


if __name__ == "__main__":
    main()