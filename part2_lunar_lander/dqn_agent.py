import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork


class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):

        # epsilon-greedy exploration
        if random.random() < self.epsilon:

            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.q_network(state_tensor)

        return torch.argmax(q_values).item()

    def update(self, replay_buffer, batch_size):

        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = (
            replay_buffer.sample(batch_size)
        )

        states = torch.FloatTensor(states).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)

        rewards = torch.FloatTensor(rewards).to(self.device)

        next_states = torch.FloatTensor(next_states).to(self.device)

        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.q_network(states)

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            ).max(1)[0]

        target_q_values = rewards + (
            self.gamma * next_q_values * (1 - dones)
        )

        loss = self.loss_fn(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )