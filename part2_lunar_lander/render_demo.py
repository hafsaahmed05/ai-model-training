"""
render_demo.py  –  Part 2: Lunar Lander DQN
Runs a small number of visual demo episodes using a trained Q-network.

Two render modes are supported:
  1. LIVE window  – renders in real time (requires a display / pygame installed)
  2. RECORD mode  – renders to an MP4 or GIF without needing a live window
                    (works headlessly on Colab or remote machines)

Outputs (record mode)
---------------------
  outputs/demo_landing.mp4  or  outputs/demo_landing.gif

Usage
-----
    # Live window (local machine with display):
    python render_demo.py --mode live

    # Record to mp4 (default, works everywhere):
    python render_demo.py --mode record

    # Record to gif instead:
    python render_demo.py --mode record --format gif

    # Run more episodes, pick a specific checkpoint:
    python render_demo.py --mode record --episodes 5 --checkpoint outputs/model_weights.pth
"""

import argparse
import os
import time

import numpy as np
import torch

from dqn_agent import DQNAgent
from env_wrappers import make_env


# ── Argument parsing ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Visual demo of trained DQN agent.")
    parser.add_argument("--checkpoint", type=str, default="outputs/model_weights.pth")
    parser.add_argument("--episodes",   type=int, default=3,
                        help="Number of demo episodes to run.")
    parser.add_argument("--max_steps",  type=int, default=1000)
    parser.add_argument("--mode",       type=str, choices=["live", "record"],
                        default="record",
                        help="'live' opens a window; 'record' saves to file.")
    parser.add_argument("--format",     type=str, choices=["mp4", "gif"],
                        default="mp4",
                        help="Output format for record mode.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--fps",        type=int, default=30,
                        help="Frames per second for the recorded output.")
    return parser.parse_args()


# ── Load agent ──────────────────────────────────────────────────────────────────
def load_agent(checkpoint_path, state_size, action_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent  = DQNAgent(state_size=state_size, action_size=action_size)
    weights = torch.load(checkpoint_path, map_location=device)
    agent.q_network.load_state_dict(weights)
    agent.q_network.eval()
    agent.epsilon = 0.0     # pure exploitation – no random actions
    print(f"[demo] Loaded weights from {checkpoint_path}  |  device: {device}")
    return agent


# ── Run one episode, collecting frames ─────────────────────────────────────────
def run_episode(agent, env, max_steps, collect_frames=False):
    """
    Run a single episode.

    Returns
    -------
    total_reward : float
    frames       : list[np.ndarray] | []  – RGB frames if collect_frames=True
    """
    state, _ = env.reset()
    total_reward = 0.0
    frames       = []
    done         = False

    for step in range(max_steps):
        # Optionally grab the rendered frame before stepping
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

        if done:
            # Grab the terminal frame too
            if collect_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            break

    return total_reward, frames


# ── LIVE mode ────────────────────────────────────────────────────────────────────
def run_live(agent, args):
    """Open a real-time window and step through episodes."""
    env = make_env(render_mode="human")

    for ep in range(1, args.episodes + 1):
        reward, _ = run_episode(agent, env, args.max_steps, collect_frames=False)
        result = "✓ landed" if reward >= 100 else "✗ failed"
        print(f"  Demo ep {ep}/{args.episodes}  |  reward: {reward:.2f}  |  {result}")
        # Small pause between episodes so the window doesn't close instantly
        time.sleep(1.0)

    env.close()
    print("[demo] Live demo complete.")


# ── RECORD mode ──────────────────────────────────────────────────────────────────
def run_record(agent, args):
    """
    Render episodes off-screen, collect RGB frames, save to mp4 or gif.
    Requires either:
      - imageio + imageio-ffmpeg  (for mp4)
      - imageio                   (for gif)
    Install with:  pip install imageio imageio-ffmpeg
    """
    try:
        import imageio
    except ImportError:
        print("[demo] imageio not found.  Install with:  pip install imageio imageio-ffmpeg")
        return

    # rgb_array mode renders frames as numpy arrays without opening a window
    env = make_env(render_mode="rgb_array")

    all_frames  = []
    all_rewards = []

    for ep in range(1, args.episodes + 1):
        reward, frames = run_episode(agent, env, args.max_steps, collect_frames=True)
        all_rewards.append(reward)
        all_frames.extend(frames)

        result = "✓ landed" if reward >= 100 else "✗ failed"
        print(f"  Recording ep {ep}/{args.episodes}  |  "
              f"reward: {reward:.2f}  |  frames: {len(frames)}  |  {result}")

    env.close()

    if not all_frames:
        print("[demo] No frames captured – check that render_mode='rgb_array' works "
              "with your gymnasium installation.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.format == "mp4":
        out_path = os.path.join(args.output_dir, "demo_landing.mp4")
        try:
            writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                                        quality=8)
            for frame in all_frames:
                writer.append_data(frame)
            writer.close()
        except Exception:
            # Fallback: use imageio v3 API
            imageio.mimwrite(out_path, all_frames, fps=args.fps)
    else:
        out_path = os.path.join(args.output_dir, "demo_landing.gif")
        imageio.mimwrite(out_path, all_frames, fps=args.fps)

    avg_reward = float(np.mean(all_rewards))
    success    = float(np.mean([r >= 100 for r in all_rewards])) * 100

    print(f"\n[demo] Video saved → {out_path}")
    print(f"[demo] {args.episodes} episodes  |  avg reward: {avg_reward:.2f}  |  "
          f"success rate: {success:.0f}%")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Spin up a temp env just to read space sizes
    tmp_env     = make_env()
    state_size  = tmp_env.observation_space.shape[0]
    action_size = tmp_env.action_space.n
    tmp_env.close()

    agent = load_agent(args.checkpoint, state_size, action_size)

    if args.mode == "live":
        run_live(agent, args)
    else:
        run_record(agent, args)


if __name__ == "__main__":
    main()