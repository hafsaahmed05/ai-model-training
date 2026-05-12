import os
import json

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# =========================
# Hyperparameters
# =========================

NUM_EPISODES = 500

MAX_STEPS = 1000

BATCH_SIZE = 64

BUFFER_CAPACITY = 100000

TARGET_UPDATE_FREQ = 10


# =========================
# Environment
# =========================

env = gym.make("LunarLander-v3")

state_size = env.observation_space.shape[0]

action_size = env.action_space.n


# =========================
# Agent + Replay Buffer
# =========================

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size
)

replay_buffer = ReplayBuffer(
    capacity=BUFFER_CAPACITY
)


# =========================
# Training
# =========================

episode_rewards = []

for episode in range(NUM_EPISODES):

    state, _ = env.reset()

    total_reward = 0

    done = False

    for step in range(MAX_STEPS):

        # select action
        action = agent.select_action(state)

        # environment step
        next_state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        done = terminated or truncated

        # store experience
        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        # train
        agent.update(
            replay_buffer,
            BATCH_SIZE
        )

        state = next_state

        total_reward += reward

        if done:
            break

    # update target network
    if episode % TARGET_UPDATE_FREQ == 0:

        agent.update_target_network()

    episode_rewards.append(total_reward)

    print(
        f"Episode {episode + 1} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {agent.epsilon:.3f}"
    )


# =========================
# Save model
# =========================
os.makedirs("outputs", exist_ok=True)

torch.save(
    agent.q_network.state_dict(),
    "outputs/model_weights.pth"
)

print("Training complete.")
print("Model saved.")

os.makedirs("outputs", exist_ok=True)
with open("outputs/training_rewards.json", "w") as f:
    json.dump({"rewards": episode_rewards}, f)