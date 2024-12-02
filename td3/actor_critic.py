import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor Network for the TD3 Agent.

    The actor network takes the state as input and outputs the action to be taken.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_size (int, optional): Number of hidden units in the network. Defaults to 128.
    """

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the layers for the actor network
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        return action


class Critic(nn.Module):
    """
    Critic Network for the TD3 Agent.

    The critic network takes the state and action as input and outputs the Q-value (estimated value) of taking that
    action in that state.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_size (int, optional): Number of hidden units in the network. Defaults to 128.
    """

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the layers for the critic network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Forward pass through the network
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class TargetNetwork:
    """
    Target Network for the TD3 Agent.

    The target network is a copy of the main network used for calculating the target Q-values during training.
    It's updated softly towards the main network parameters.

    Args:
        model (nn.Module): The model to create a target network for (Actor or Critic).
        tau (float, optional): Soft update parameter (0 < tau <= 1). Defaults to 0.005.
    """

    def __init__(self, model, tau=0.005):
        self.model = model
        self.state_dim = model.state_dim
        self.action_dim = model.action_dim
        self.target = self.clone_network(model)
        self.tau = tau

    def clone_network(self, model):
        """
         Creates a new instance of the model with the same arguments.
        """
        if isinstance(model, Actor):
            target = Actor(self.state_dim, self.action_dim)
        elif isinstance(model, Critic):
            target = Critic(self.state_dim, self.action_dim)

        target.load_state_dict(model.state_dict())
        return target

    def update(self):
        """
        Soft update of the target network towards the main network parameters.
        """
        for target_param, param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)

    def get_target(self):
        """
        Returns the target network.
        """
        return self.target
