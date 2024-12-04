import os
import tqdm
import gym
import numpy as np
import pickle
import torch
from ppo_agent import PPOAgent
from ppo_replay_buffer import PPOReplayBuffer
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

if __name__ == "__main__":
    """
    Main script to train the PPO agent on the Mars Explorer environment.

    This script defines training parameters, initializes the environment and agent,
    collects experience, trains the agent, and saves the results.
    """

    # Define directories for saving results
    results_path = "training_results"
    plot_path = "plot_figs"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Environment parameters
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)
    state_dim = np.prod(env.observation_space.shape)  # Flattened state space
    action_dim = env.action_space.n  # Discrete action space (4 actions)

    batch_size = 64
    epochs = 10
    num_episodes = 150
    replay_buffer_size = 100000

    # Initialize PPO agent and replay buffer
    agent = PPOAgent(state_dim, action_dim)
    replay_buffer = PPOReplayBuffer(replay_buffer_size, state_dim, action_dim)

    # Storage for results
    rewards_per_episode = []
    steps_per_episode = []

    # Training loop
    pbar = tqdm.trange(num_episodes, desc="Training PPO")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.store(state, action, log_prob, reward, done,value)
            episode_reward += reward
            episode_steps += 1

            state = next_state

            if done or episode_steps >= conf.max_steps:
                # Finish trajectory and update PPO
                last_value = 0 if done else agent.value_net(torch.tensor(next_state, dtype=torch.float32)).item()
                replay_buffer.compute_advantages_and_returns(last_value)

                # Train the PPO agent using the replay buffer
                agent.train(replay_buffer, epochs, batch_size)

                # Log results
                rewards_per_episode.append(episode_reward)
                steps_per_episode.append(episode_steps)
                break

    # Save final results
    with open(os.path.join(results_path, "ppo_results.pkl"), "wb") as f:
        pickle.dump({
            "ppo_rewards": rewards_per_episode,
            "ppo_steps": steps_per_episode,
        }, f)

    print(f"Training complete. Results saved in {results_path}.")

    env.close()
