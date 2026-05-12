import random
from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(experience)

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):

        return len(self.buffer)
    

if __name__ == "__main__":

    buffer = ReplayBuffer(capacity=1000)

    for i in range(10):

        state = np.random.randn(8)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(8)
        done = np.random.choice([True, False])

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    print("Buffer size:", len(buffer))

    sample = buffer.sample(batch_size=4)

    print("States shape:", sample[0].shape)
    print("Actions shape:", sample[1].shape)