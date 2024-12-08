import os

import torch
import torch.optim as optim
import numpy as np
from ppo_memory_buffer import MemoryBuffer
import torch.nn.functional as F

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for training on the MarsExplorer environment.
    """
    def __init__(self, policy_network, state_space, action_space, lr=3e-4, gamma=0.99, eps_clip=0.2, gae_lambda=0.95):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)

    def select_action(self, state):
        """
        Select an action based on the current state using the policy network.

        Args:
            state (np.ndarray): The current state of the environment.
        Returns:
            action (int): The action to take.
            action_prob (float): The probability of the action.
            state_value (float): The state value from the critic.
        """
        state_tensor = torch.FloatTensor(state).view(-1, self.policy_network.input_size).to(self.device)
        logits, value = self.policy_network(state_tensor)

        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        return action.item(), action_probs[0, action.item()].item(), value.item()

    def compute_gae(self, rewards, values, dones):
            """
            Compute Generalized Advantage Estimation (GAE).

            Args:
                rewards (list): List of rewards collected during an episode.
                values (list): List of state values from the critic.
                dones (list): List of done flags indicating episode termination.
            Returns:
                advantages (np.ndarray): Computed advantages.
                returns (np.ndarray): Discounted returns.
            """
            advantages = np.zeros_like(rewards, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)

            gae = 0
            for t in reversed(range(len(rewards))):
                # Handle next_value to avoid out-of-bounds access
                next_value = 0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            return advantages, returns

    def update_policy(self, trajectories):
        """
        Update the policy network using collected trajectories.

        Args:
            trajectories (dict): A dictionary containing states, actions, rewards, etc.
        """
        states = torch.tensor(trajectories['states'], dtype=torch.float32, device=self.device).view(-1,  self.state_space)
        actions = torch.tensor(trajectories['actions'], dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(trajectories['advantages'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32, device=self.device)

        # Normalize advantages
        # Only normalize if the standard deviation is non-zero
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        actor_losses = []
        critic_losses = []

        # Perform optimization steps
        for _ in range(10):
            logits, values = self.policy_network(states)
            action_probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)

            log_probs = action_dist.log_prob(actions)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return actor_losses, critic_losses


    def train(self, env, num_episodes, max_timesteps, update_interval=10):
        """
        Train the PPO agent on the environment.

        Args:
            env (gym.Env): The MarsExplorer environment.
            num_episodes (int): The number of episodes to train the agent.
            max_timesteps (int): Maximum timesteps per episode.
            update_interval (int): Number of timesteps between policy updates.
        """
        # Defining the path to store models
        model_path = os.path.join("trained_models")

        # Create the model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        memory = MemoryBuffer()

        rewards_list = []
        actor_losses_list = []
        steps_list = []

        for episode in range(num_episodes):
            state, _ = env.reset()  # Reset the environment to get the initial state
            episode_rewards = 0
            done = False
            timestep = 0
            episode_actor_losses = []  # Track losses for the current episode

            while not done and timestep < max_timesteps:
                # env.render()
                # Select action and store experience
                action, action_prob, value = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                memory.store(state, action, reward, done, action_prob, value)

                state = next_state
                episode_rewards += reward
                timestep += 1

                # If it's time to update the policy
                if timestep % update_interval == 0 or done:
                    # Calculate advantages and returns
                    trajectories = memory.get_trajectories()
                    advantages, returns = self.compute_gae(trajectories['rewards'], trajectories['values'],
                                                           trajectories['dones'])
                    trajectories['advantages'] = advantages
                    trajectories['returns'] = returns

                    # Update policy
                    actor_losses, critic_losses = self.update_policy(trajectories)
                    episode_actor_losses.extend(actor_losses)
                    memory.clear()

            rewards_list.append(episode_rewards)
            steps_list.append(timestep)
            actor_losses_list.append(np.mean(episode_actor_losses))  # Store mean actor loss for the episode
            # print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_rewards}")

        # Save the model
        ppo_policy_network_path = model_path + "/ppo_policy_network.pth"
        torch.save(self.policy_network.state_dict(), ppo_policy_network_path)
        print(f"Model saved as ppo_policy_network.pth -> {ppo_policy_network_path}")

        return rewards_list, actor_losses_list, steps_list