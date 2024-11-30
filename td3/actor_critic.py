import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        Initialize the Actor network.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            max_action (float): Maximum value of the action.
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        """
        Forward pass for the Actor network.
        Args:
            state (torch.Tensor): Input state.
        Returns:
            torch.Tensor: Scaled action output.
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Critic networks (Q1 and Q2).
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_l2 = nn.Linear(256, 256)
        self.q1_l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.q2_l1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_l2 = nn.Linear(256, 256)
        self.q2_l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        Forward pass for both Q1 and Q2 networks.
        Args:
            state (torch.Tensor): Input state.
            action (torch.Tensor): Input action.
        Returns:
            tuple: Q1 and Q2 values.
        """
        sa = torch.cat([state, action], dim=1)

        # Q1 computation
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        # Q2 computation
        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def q1(self, state, action):
        """
        Compute only Q1 value.
        Args:
            state (torch.Tensor): Input state.
            action (torch.Tensor): Input action.
        Returns:
            torch.Tensor: Q1 value.
        """
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        return q1
