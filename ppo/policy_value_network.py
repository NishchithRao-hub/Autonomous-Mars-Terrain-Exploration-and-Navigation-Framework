import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    """
    Policy network for PPO.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output actions.
        hidden_size (int): Number of hidden units in the layers.
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Log standard deviation for stochastic policy

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = self.log_std.exp().expand_as(mean)  # Convert log_std to standard deviation
        return mean, std

    def act(self, state):
        """
        Sample an action from the policy's distribution.
        Args:
            state (torch.Tensor): The current state.
        Returns:
            action (torch.Tensor): Sampled action.
            log_prob (torch.Tensor): Log probability of the action.
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        """
        Evaluate the log probability of given actions.
        Args:
            state (torch.Tensor): The current state.
            action (torch.Tensor): Actions to evaluate.
        Returns:
            log_prob (torch.Tensor): Log probability of the action.
            entropy (torch.Tensor): Entropy of the distribution.
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class ValueNetwork(nn.Module):
    """
    Value network for PPO.

    Args:
        state_dim (int): Dimension of the input state.
        hidden_size (int): Number of hidden units in the layers.
    """
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
