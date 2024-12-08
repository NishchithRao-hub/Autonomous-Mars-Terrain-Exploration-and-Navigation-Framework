import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    The policy network defines the architecture for both the policy (actor) and the value (critic) heads.
    """

    def __init__(self, input_size, action_space):
        super(PolicyNetwork, self).__init__()

        self.input_size = input_size
        self.action_space = action_space

        # Fully connected layers for processing 1D input
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Actor head
        self.actor = nn.Linear(128, action_space)

        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.
        Returns:
            logits (torch.Tensor): Action logits for the policy (actor head).
            value (torch.Tensor): State value estimation (critic head).
        """

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor and Critic heads
        logits = self.actor(x)
        value = self.critic(x)

        return logits, value
