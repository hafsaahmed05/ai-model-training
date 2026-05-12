import gymnasium as gym
import numpy as np


def make_env(render_mode=None):
    """
    Create LunarLander environment.
    """
    env = gym.make(
        "LunarLander-v3",
        render_mode=render_mode
    )

    return env


if __name__ == "__main__":

    env = make_env()

    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    print("Observation:", obs)

    print("Action space:", env.action_space)
    print("Number of actions:", env.action_space.n)

    done = False
    total_reward = 0

    while not done:

        # random action for testing
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        print(f"Action: {action}")
        print(f"Reward: {reward:.2f}")

    print("Episode finished")
    print("Total reward:", total_reward)

    env.close()