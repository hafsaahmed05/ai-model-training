import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):

        return self.network(x)
    
if __name__ == "__main__":

    state_size = 8
    action_size = 4

    model = QNetwork(state_size, action_size)

    sample_state = torch.randn(1, state_size)

    q_values = model(sample_state)

    print("Input shape:", sample_state.shape)
    print("Output shape:", q_values.shape)
    print("Q-values:", q_values)