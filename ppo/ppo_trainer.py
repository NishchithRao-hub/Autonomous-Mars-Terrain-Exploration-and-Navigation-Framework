# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions import Categorical
# import numpy as np
# import gym
# from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
#
#
# class PPO(nn.Module):
#     def __init__(self, obs_space, action_space, hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
#                  value_loss_coef=0.5, entropy_coef=0.01):
#         super(PPO, self).__init__()
#
#         self.gamma = gamma
#         self.clip_epsilon = clip_epsilon
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#
#         # Policy network
#         self.policy = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # First Conv2D expects 1 input channel
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * 21 * 21, 128),
#             nn.ReLU(),
#             nn.Linear(128, 4)  # Output layer for 4 discrete actions
#         )
#
#         # Value network
#         self.value = nn.Sequential(
#             nn.Conv2d(obs_space[2], 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(obs_space[0] * obs_space[1] * 32, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         # Optimizer
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#
#     def forward(self, state):
#         action_logits = self.policy(state)
#         state_values = self.value(state)
#         return action_logits, state_values
#
#     def select_action(self, state):
#         action_logits, _ = self(state)
#         distribution = Categorical(F.softmax(action_logits, dim=-1))
#         action = distribution.sample()
#         return action, distribution.log_prob(action), distribution.entropy()
#
#     def compute_gae(self, rewards, masks, values, next_values, gamma, lambda_):
#         deltas = rewards + gamma * next_values * masks - values
#         advantages = torch.zeros_like(rewards)
#         gae = 0
#         for t in reversed(range(len(rewards))):
#             gae = deltas[t] + gamma * lambda_ * masks[t] * gae
#             advantages[t] = gae
#         return advantages
#
#     def update(self, states, actions, log_probs, returns, advantages):
#         action_logits, values = self(states)
#
#         distribution = Categorical(F.softmax(action_logits, dim=-1))
#         new_log_probs = distribution.log_prob(actions)
#         entropy = distribution.entropy().mean()
#
#         # Compute ratio (pi_theta / pi_theta_old)
#         ratios = torch.exp(new_log_probs - log_probs)
#
#         # Compute surrogate losses
#         surr1 = ratios * advantages
#         surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
#
#         # Losses
#         policy_loss = -torch.min(surr1, surr2).mean()
#         value_loss = F.mse_loss(values.flatten(), returns)
#         total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
#
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()
#
#
# class PPOAgent:
#     def __init__(self, env, gamma=0.99, lambda_=0.95, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01,
#                  hidden_dim=64, lr=3e-4, num_epochs=10):
#         self.env = env
#         self.gamma = gamma
#         self.lambda_ = lambda_
#         self.clip_epsilon = clip_epsilon
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#         self.num_epochs = num_epochs
#
#         # Initialize PPO model
#         self.model = PPO(env.observation_space.shape, env.action_space, hidden_dim, lr, gamma, clip_epsilon,
#                          value_loss_coef, entropy_coef)
#
#     def train(self, num_episodes=1000, max_steps=400):
#         for episode in range(num_episodes):
#             states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []
#             state = self.env.reset()
#
#             for step in range(max_steps):
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Adds batch and channel dimensions
#                 # Remove the extra dimension (1) in the last place
#                 state_tensor = state_tensor.squeeze(-1)
#                 # Assuming the original state_tensor is [1, 21, 21, 1]
#                 state_tensor = state_tensor.permute(0, 3, 2, 1)  # Change shape to [1, 1, 21, 21]
#                 print(state_tensor.shape)
#                 action, log_prob, _ = self.model.select_action(state_tensor)
#                 next_state, reward, done, _ = self.env.step(action.item())
#
#                 states.append(state_tensor)
#                 actions.append(action)
#                 log_probs.append(log_prob)
#                 rewards.append(reward)
#                 masks.append(1 - done)
#                 values.append(self.model.value(state_tensor).flatten())
#
#                 state = next_state
#
#                 if done:
#                     break
#
#             # Compute returns and advantages
#             next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
#             _, next_value = self.model(next_state_tensor)
#             next_value = next_value.flatten()
#
#             returns = self.compute_returns(rewards, masks, next_value)
#             advantages = self.model.compute_gae(rewards, masks, values, next_value, self.gamma, self.lambda_)
#
#             # Convert lists to tensors
#             states = torch.cat(states)
#             actions = torch.tensor(actions)
#             log_probs = torch.cat(log_probs)
#             returns = torch.tensor(returns)
#             advantages = torch.tensor(advantages)
#
#             # Update PPO model
#             for _ in range(self.num_epochs):
#                 self.model.update(states, actions, log_probs, returns, advantages)
#
#     def compute_returns(self, rewards, masks, next_value):
#         returns = torch.zeros_like(rewards)
#         gae = 0
#         for t in reversed(range(len(rewards))):
#             gae = rewards[t] + self.gamma * next_value * masks[t] - values[t]
#             returns[t] = gae
#             next_value = values[t]
#         return returns
#
#
# def get_conf():
#     conf["size"] = [30, 30]
#     conf["obstacles"] = 20
#     conf["lidar_range"] = 4
#     conf["obstacle_size"] = [1, 3]
#
#     conf["viewer"]["night_color"] = (0, 0, 0)
#     conf["viewer"]["draw_lidar"] = True
#
#     conf["viewer"]["drone_img"] = "mars-explorer/tests/img/drone.png"
#     conf["viewer"]["obstacle_img"] = "mars-explorer/tests/img/block.png"
#     conf["viewer"]["background_img"] = "mars-explorer/tests/img/mars.jpg"
#     conf["viewer"]["light_mask"] = "mars-explorer/tests/img/light_350_hard.png"
#     return conf
#
#
# # Example usage
# if __name__ == "__main__":
#     # Create environment
#     env = gym.make('mars_explorer:exploConf-v1', conf=conf)
#
#     # Initialize and train agent
#     agent = PPOAgent(env)
#     agent.train(num_episodes=10)
import gym
import torch
import numpy as np
from tqdm import tqdm
from config import config
from ppo import PPO
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf



class PPOTrainer:
    def __init__(self, env, ppo, config):
        self.env = env
        self.ppo = ppo
        self.max_steps = config['max_steps']
        self.batch_size = config['batch_size']
        self.num_episodes = config['num_episodes']

    def collect_trajectory(self):
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        next_values = []

        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.ppo.policy.act(state_tensor)
            next_state, reward, done, _ = self.env.step(action)

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(state_tensor)  # Simplified, use actual value prediction here
            dones.append(done)

            total_reward += reward
            state = next_state

        # Compute next values (for GAE)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_value = self.ppo.policy.critic(next_state_tensor)
        next_values.extend([next_value] * len(states))

        return np.array(states), np.array(actions), np.array(rewards), np.array(values), np.array(dones), np.array(next_values), total_reward

    def train(self):
        for episode in tqdm(range(self.num_episodes)):
            states, actions, rewards, values, dones, next_values, total_reward = self.collect_trajectory()

            # Update the policy
            policy_loss, value_loss = self.ppo.update(states, actions, rewards, values, dones, next_values)

            print(f"Episode {episode}, Total Reward: {total_reward}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")


if __name__ == "__main__":

    # config_with_size = config.copy()
    # config_with_size['size'] = [10,10]

    env = gym.make('mars_explorer:exploConf-v1', conf=conf)
    # ppo = PPO(obs_space=(env.sizeX, env.sizeY, 1), action_space=env.action_space.n, config=config)
    ppo = PPO(obs_space=env.observation_space, action_space=env.action_space.n, config=config)
    trainer = PPOTrainer(env, ppo, config)
    trainer.train()
