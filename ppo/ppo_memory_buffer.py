import numpy as np


class MemoryBuffer:
    """
    Memory buffer for storing and preparing trajectories for PPO updates.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        """
        Store a single timestep of data into the buffer.

        Args:
            state (np.ndarray): The state of the environment.
            action (int): The action taken.
            reward (float): The reward received.
            done (bool): Whether the episode is done.
            log_prob (float): Log probability of the action taken.
            value (float): Value estimate of the state.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """
        Clear all stored data from the buffer.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_trajectories(self):
        """
        Retrieve and format stored trajectories for training.

        Returns:
            dict: A dictionary containing the trajectories.
        """
        trajectories = {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
        }
        self.clear()  # Clear the buffer after retrieving data
        return trajectories
