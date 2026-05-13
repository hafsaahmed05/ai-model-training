import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# ==========================================
# Hyperparameters
# ==========================================

NUM_EPISODES = 500

MAX_STEPS = 1000

BATCH_SIZE = 64

BUFFER_CAPACITY = 100000

TARGET_UPDATE_FREQ = 10


# ==========================================
# Environment
# ==========================================

env = gym.make("LunarLander-v3")

state_size = env.observation_space.shape[0]

action_size = env.action_space.n


# ==========================================
# Agent + Replay Buffer
# ==========================================

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size
)

replay_buffer = ReplayBuffer(
    capacity=BUFFER_CAPACITY
)


# ==========================================
# Output Directory
# ==========================================

os.makedirs("outputs/modified", exist_ok=True)


# ==========================================
# Training
# ==========================================

episode_rewards = []

for episode in range(NUM_EPISODES):

    state, _ = env.reset()

    total_reward = 0

    done = False

    for step in range(MAX_STEPS):

        # Select action
        action = agent.select_action(state)

        # Environment step
        next_state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        done = terminated or truncated

        # ==========================================
        # MODIFICATION EXPERIMENT
        # Extra penalty for drifting from center
        # ==========================================

        reward -= abs(next_state[0]) * 0.1

        # Store transition
        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train agent
        agent.update(
            replay_buffer,
            BATCH_SIZE
        )

        state = next_state

        total_reward += reward

        if done:
            break

    # Update target network
    if episode % TARGET_UPDATE_FREQ == 0:

        agent.update_target_network()

    episode_rewards.append(total_reward)

    print(
        f"Episode {episode + 1} | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {agent.epsilon:.3f}"
    )


env.close()


# ==========================================
# Save Model + Rewards
# ==========================================

torch.save(
    agent.q_network.state_dict(),
    "outputs/modified/model_weights.pth"
)

np.save(
    "outputs/modified/rewards.npy",
    np.array(episode_rewards)
)

print("\nTraining complete.")
print("Modified model saved.")


# ==========================================
# Simple Evaluation
# ==========================================

print("\n===== Evaluation =====")

agent.epsilon = 0.0

TEST_EPISODES = 10

test_rewards = []

test_env = gym.make("LunarLander-v3")

for episode in range(TEST_EPISODES):

    state, _ = test_env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = (
            test_env.step(action)
        )

        done = terminated or truncated

        state = next_state

        total_reward += reward

    test_rewards.append(total_reward)

    print(
        f"Test Episode {episode + 1}: "
        f"{total_reward:.2f}"
    )

test_env.close()

print(
    "\nAverage Test Reward:",
    np.mean(test_rewards)
)


# ==========================================
# Reward Curve Plot
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.title("Modified DQN Training Rewards")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid(True)

plt.savefig(
    "outputs/modified/reward_curve.png"
)

plt.show()