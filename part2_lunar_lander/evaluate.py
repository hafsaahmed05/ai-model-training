"""
evaluate.py  –  Part 2: Lunar Lander DQN
Loads a trained Q-network and runs pure exploitation episodes (epsilon = 0).

Outputs
-------
- Prints average reward and success rate over N evaluation episodes
- Saves outputs/eval_rewards.json   (raw per-episode rewards for further analysis)
- Saves outputs/eval_summary.txt    (human-readable summary for the report)

Usage
-----
    python evaluate.py
    python evaluate.py --checkpoint outputs/model_weights.pth --episodes 100
"""

import argparse
import json
import os

import numpy as np
import torch

from dqn_agent import DQNAgent
from env_wrappers import make_env


# ── Argument parsing ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN on LunarLander.")
    parser.add_argument("--checkpoint", type=str, default="outputs/model_weights.pth",
                        help="Path to saved Q-network weights.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max steps per episode.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation (slow).")
    return parser.parse_args()


# ── Success heuristic ───────────────────────────────────────────────────────────
def is_successful_landing(total_reward: float) -> bool:
    """
    Gymnasium's LunarLander awards +100 for a successful landing on the pad
    and deducts points for crashes / drifting.  Episodes with reward >= 200
    are considered 'solved'; we use a more lenient threshold of >= 100 to
    count as a safe landing for the success-rate metric.
    """
    return total_reward >= 100.0


# ── Core evaluation loop ────────────────────────────────────────────────────────
def run_evaluation(agent, env, num_episodes, max_steps):
    """
    Run `num_episodes` with epsilon fixed at 0 (pure greedy).

    Returns
    -------
    rewards : list[float]  – total reward per episode
    """
    agent.epsilon = 0.0          # no exploration during evaluation
    rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        status = "✓ landed" if is_successful_landing(total_reward) else "✗ failed"
        print(f"  Eval ep {ep:>3d}/{num_episodes}  |  reward: {total_reward:>8.2f}  |  {status}")

    return rewards


# ── Print + save summary ────────────────────────────────────────────────────────
def summarise(rewards, output_dir):
    avg_reward   = float(np.mean(rewards))
    std_reward   = float(np.std(rewards))
    min_reward   = float(np.min(rewards))
    max_reward   = float(np.max(rewards))
    success_rate = float(np.mean([is_successful_landing(r) for r in rewards])) * 100.0

    lines = [
        "=" * 50,
        "  DQN Evaluation Summary",
        "=" * 50,
        f"  Episodes evaluated : {len(rewards)}",
        f"  Average reward     : {avg_reward:.2f}",
        f"  Std deviation      : {std_reward:.2f}",
        f"  Min reward         : {min_reward:.2f}",
        f"  Max reward         : {max_reward:.2f}",
        f"  Success rate (>=100): {success_rate:.1f}%",
        "=" * 50,
    ]

    # Rubric thresholds
    if avg_reward >= 200:
        lines.append("  [STAR] Stretch goal reached (>=200 avg reward)!")
    elif avg_reward >= 100:
        lines.append("  [PASS] Minimum performance threshold met (>=100 avg reward).")
    else:
        lines.append("  [FAIL] Below minimum threshold -- see report analysis.")

    summary_text = "\n".join(lines)
    print("\n" + summary_text + "\n")

    # Save plain-text summary (utf-8 so special chars don't break on Windows)
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "eval_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"[evaluate] Summary saved → {txt_path}")

    # Save raw rewards as JSON (used by visualize.py)
    json_path = os.path.join(output_dir, "eval_rewards.json")
    with open(json_path, "w") as f:
        json.dump({
            "rewards":      rewards,
            "avg_reward":   avg_reward,
            "std_reward":   std_reward,
            "success_rate": success_rate,
        }, f, indent=2)
    print(f"[evaluate] Raw rewards saved → {json_path}")

    return avg_reward, success_rate


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")
    print(f"[evaluate] Loading weights from: {args.checkpoint}")

    # Build env
    render_mode = "human" if args.render else None
    env = make_env(render_mode=render_mode)

    state_size  = env.observation_space.shape[0]   # 8
    action_size = env.action_space.n               # 4

    # Build agent and load weights
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.q_network.load_state_dict(checkpoint)
    agent.q_network.eval()
    print(f"[evaluate] Weights loaded. Running {args.episodes} evaluation episodes…\n")

    # Run evaluation
    rewards = run_evaluation(agent, env, args.episodes, args.max_steps)
    env.close()

    # Print and save summary
    summarise(rewards, args.output_dir)


if __name__ == "__main__":
    main()