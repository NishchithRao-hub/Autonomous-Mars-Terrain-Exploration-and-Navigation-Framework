import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from actor_critic import Actor, Critic, TargetNetwork
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os


class TD3Agent:
    """
    TD3 Agent for Reinforcement Learning.

    This class implements the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm
    for training an agent in a continuous action space environment.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_size (int, optional): Number of hidden units in the networks. Defaults to 128.
        gamma (float, optional): Discount factor (0 < gamma < 1). Defaults to 0.99.
        tau (float, optional): Soft update parameter for target networks (0 < tau <= 1). Defaults to 0.005.
        policy_noise (float, optional): Exploration noise added to the selected action during training. Defaults to 0.2.
        policy_freq (int, optional): Frequency of updating the actor network (update every `policy_freq` iterations).
        Defaults to 2.
        learning_rate (float, optional): Learning rate for the optimizers. Defaults to 1e-3.
    """

    def __init__(self, state_dim, action_dim, batch_size=64, hidden_size=128, gamma=0.99, tau=0.01, policy_noise=0.2,
                 policy_freq=2, learning_rate=1e-4):
        # Initialize the actor and critic networks
        self.batch_size = batch_size
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size)

        # Initialize target networks
        self.actor_target = TargetNetwork(self.actor, tau)
        self.critic_1_target = TargetNetwork(self.critic_1, tau)
        self.critic_2_target = TargetNetwork(self.critic_2, tau)

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)

        # Hyperparameters for TD3
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=1000000, batch_size=self.batch_size)

        # For action noise
        self.action_dim = action_dim

        # Tracking losses
        self.last_critic_loss = 0
        self.last_actor_loss = 0

        # For tracking updates
        self.timestep = 0

    def select_action(self, state):
        """
        Select an action based on the current state using the actor's policy with exploration noise.

        Args:
            state (np.array): Current state of the environment.

        Returns:
            tuple: A tuple containing the selected action and the associated action values.
        """
        state = torch.FloatTensor(state).view(-1, self.actor.state_dim).unsqueeze(0)
        action_values = self.actor(state).detach().numpy()[0]
        action = np.argmax(action_values)
        return action, action_values

    def update(self):
        """
       Update the actor and critic networks using the experiences from the replay buffer.
       """
        # Sample a batch of experiences from the replay buffer
        if len(self.replay_buffer.buffer) >= self.batch_size:
            state, action, action_values, reward, next_state, done = self.replay_buffer.sample()
        else:
            return

        # Convert to pytorch tensor
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        action_values = torch.FloatTensor(action_values)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)

        # Reshape the tensor to pass to Neural Networks
        state = state.view(64, -1)
        action = action.view(64, -1)
        action_values = action_values.view(64, -1)
        reward = reward.view(64, -1)
        next_state = next_state.view(64, -1)
        done = done.view(64, -1)

        # Update critic networks
        noise = torch.randn_like(action) * self.policy_noise
        noise = noise.clamp(-0.5, 0.5)
        next_action = self.actor_target.get_target()(next_state) + noise
        # next_action = self.actor_target.get_target()(next_state) + torch.randn_like(action) * self.policy_noise

        # Get the Q-value from the target critics
        target_q1 = self.critic_1_target.get_target()(next_state, next_action)
        target_q2 = self.critic_2_target.get_target()(next_state, next_action)

        target_q1 = target_q1.detach()
        target_q2 = target_q2.detach()

        # Take the minimum Q-value between the two critics for stability
        target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        # Get the current Q-values from the critic networks
        current_q1 = self.critic_1(state, action_values)
        current_q2 = self.critic_2(state, action_values)

        # Compute critic loss and update critic networks
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Store the critic loss
        self.last_critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2

        # Update the actor network every `policy_freq` iterations
        if self.timestep % self.policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self.actor_target.update()
            self.critic_1_target.update()
            self.critic_2_target.update()

            # Store the actor loss
            self.last_actor_loss = actor_loss.item()

        self.timestep += 1

    def train(self, episodes):
        """
        Train the TD3 Agent over several episodes.

        This method trains the TD3 agent by interacting with the environment, collecting experiences,
        and updating the actor and critic networks using the replay buffer.

        Args:
            episodes (int): Number of episodes to train the agent for.

        Returns:
            tuple: A tuple containing three lists:
                - train_rewards: List of average episode rewards across all episodes.
                - train_actor_loss: List of average actor network losses across all updates.
                - train_critic_loss: List of average critic network losses (combined) across all updates.
        """

        # Defining the path to store models
        model_path = os.path.join("trained_models")

        # Create the model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        train_rewards = []
        train_actor_loss = []
        train_critic_loss = []

        """ Train the TD3 agent over several episodes """
        for episode in range(episodes):
            state = self.reset_env()
            ep_reward = 0
            critic_loss = 0
            actor_loss = 0
            done = False

            while not done:
                # self.env.render() # Uncomment to render the environment during training
                action, action_values = self.select_action(state)  # Get action and action values from actor
                next_state, reward, done, info = self.step_env(action)  # Take step in environment

                self.replay_buffer.add(state, action, action_values, reward, next_state,
                                       done)  # Add experience to buffer
                state = next_state  # Update state

                if len(self.replay_buffer.buffer) >= self.batch_size:
                    self.update()  # Update the networks

                ep_reward += reward
                critic_loss += self.last_critic_loss
                actor_loss += self.last_actor_loss

            train_rewards.append(ep_reward)
            train_actor_loss.append(actor_loss)
            train_critic_loss.append(critic_loss)

        # Save the model after training
        self.save_model(model_path + "/td3_actor.pth", model_path + "/td3_critic_1.pth", model_path + "/td3_critic_2.pth")
        return train_rewards, train_actor_loss, train_critic_loss

    def reset_env(self):
        # Reset the environment
        self.env = gym.make('mars_explorer:exploConf-v1', conf=conf)  # Initialize the environment
        state = self.env.reset()  # Reset and get the initial state
        return state

    def step_env(self, action):
        # Step the environment here (using your MarsExplorer environment)
        next_state, reward, done, info = self.env.step(action)
        # Return the next state, reward, done, and info
        return next_state, reward, done, info

    def save_model(self, actor_path, critic_1_path, critic_2_path):
        """
        Save the actor and critic models to the specified paths
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)
        print(f"Models saved: Actor -> {actor_path}, Critic_1 -> {critic_1_path}, Critic_2 -> {critic_2_path}")