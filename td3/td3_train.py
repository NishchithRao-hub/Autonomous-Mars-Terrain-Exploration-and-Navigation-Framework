import gym
import numpy as np
import torch
from td3.runners.td3_agent import TD3Agent

def train_td3(env_name, episodes=1000, max_steps=500, batch_size=256, device='cpu'):
    # Initialize environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize TD3 Agent
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device
    )

    rewards_history = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, noise=0.1)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            agent.add_to_replay_buffer(state, action, next_state, reward, done)

            # Train the agent
            agent.train(batch_size=batch_size)

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

        # Optionally save models every N episodes
        if (episode + 1) % 50 == 0:
            torch.save(agent.actor.state_dict(), f"actor_{episode + 1}.pth")
            torch.save(agent.critic1.state_dict(), f"critic1_{episode + 1}.pth")
            torch.save(agent.critic2.state_dict(), f"critic2_{episode + 1}.pth")

    env.close()
    return rewards_history

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="mars_explorer:exploConf-v1",
                        help="Environment name")
    parser.add_argument('--episodes', type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument('--max_steps', type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to run the training on ('cpu' or 'cuda')")
    args = parser.parse_args()

    rewards = train_td3(
        env_name=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device
    )

    # Save rewards for visualization later
    np.save("rewards_history.npy", rewards)
