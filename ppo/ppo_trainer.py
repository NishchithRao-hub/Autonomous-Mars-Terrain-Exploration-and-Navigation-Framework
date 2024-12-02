import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (batch_size, 441)
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input (batch_size, 441)
        return self.fc(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, critic_coef=0.5, entropy_coef=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards, dones, values):
        returns = []
        G = 0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            G = r + (1 - d) * self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self, states, actions, log_probs, rewards, dones):
        # Convert to tensors with requires_grad=False for inputs
        states = torch.FloatTensor(np.array(states))  # No need for requires_grad
        actions = torch.LongTensor(actions)  # No need for requires_grad
        old_log_probs = torch.FloatTensor(log_probs)  # No need for requires_grad
        rewards = torch.FloatTensor(rewards)  # No need for requires_grad
        dones = torch.FloatTensor(dones)  # No need for requires_grad

        # Get value estimates (no gradient tracking for states)
        values = self.value(states)  # Shape (batch_size, 1)

        returns = self.compute_returns(rewards, dones, values.detach())  # Detach values (no gradient tracking)
        advantages = returns - values.detach().squeeze()  # Ensure correct shape for value

        # Update policy
        for _ in range(4):  # Multiple epochs
            new_probs = self.policy(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            (policy_loss - self.entropy_coef * entropy.mean()).backward()  # Policy loss with entropy
            self.policy_optimizer.step()

        # Update value function
        value_loss = ((returns - values.detach().squeeze()) ** 2).mean()  # Value loss (use detach for values)
        self.value_optimizer.zero_grad()
        value_loss.backward()  # Backpropagate through value network
        self.value_optimizer.step()


env = gym.make('mars_explorer:exploConf-v1', conf=conf)
state_dim = 21 * 21  # Flatten the state (21x21)
action_dim = env.action_space.n

ppo_agent = PPOAgent(state_dim, action_dim)

n_episodes = 1000
max_steps = 500
batch_size = 64

for episode in range(n_episodes):
    state = env.reset()
    states, actions, rewards, dones, log_probs = [], [], [], [], []
    episode_reward = 0

    for t in range(max_steps):
        action, log_prob = ppo_agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        state = next_state

        episode_reward += reward
        if done:
            break

    ppo_agent.update(states, actions, log_probs, rewards, dones)

    print(f"Episode {episode}, Reward: {episode_reward}")


state = env.reset()
done = False
while not done:
    action, _ = ppo_agent.get_action(state)
    state, _, done, _ = env.step(action)
    env.render()
