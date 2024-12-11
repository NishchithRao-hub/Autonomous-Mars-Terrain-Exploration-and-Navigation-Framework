from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
import tqdm
import pickle

if __name__ == "__main__":
    """
    Main script to train and evaluate the PPO agent on the Mars Explorer environment.

    This script defines the training parameters, initializes the environment and agent,
    runs multiple training trials, and plots the average performance of the agent.
    """

    results_path = os.path.join("training_results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Initialize the environment
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)  # Initialize the environment
    state_dim = np.prod(env.observation_space.shape)  # Flattened state space (21x21 grid)
    action_dim = env.action_space.n  # 4 discrete actions (up, down, left, right)

    # Training parameters
    num_trials = 3  # Number of trials
    episodes_per_trial = 100  # Episodes per training trial

    # Initialize storage for results
    baseline_ppo_returns = []

    """
        Train Stable Baselines3 PPO Agent
    """
    # Vectorize the environment for Stable Baselines3
    stable_env = make_vec_env(lambda: gym.make('mars_explorer:exploConf-v1', conf=conf), n_envs=1)

    # Set up Stable Baselines3 PPO
    stable_ppo_agent = PPO(
        "MlpPolicy",  # Use a Multi-Layer Perceptron policy
        stable_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps to run for each environment per update
        batch_size=64,  # Batch size for minibatch optimization
        n_epochs=10,  # Number of epochs to optimize the policy
        gamma=0.99,  # Discount factor
        clip_range=0.2,  # Clipping range for PPO
        gae_lambda=0.95,  # GAE lambda
    )

    """
    Train the PPO agent for multiple trials and collect performance data.
    """
    tr_bar = tqdm.trange(num_trials, desc="Training Stable Baselines3 PPO")
    max_steps = conf.get('max_steps', 800)  # Use a default if not set

    for trial in tr_bar:
        stable_ppo_agent.learn(total_timesteps=episodes_per_trial * max_steps)

        rewards = []
        obs = stable_env.reset()
        for _ in range(episodes_per_trial):
            action, _ = stable_ppo_agent.predict(obs, deterministic=True)
            obs, reward, done, info = stable_env.step(action)
            rewards.append(reward)

            if done:
                obs = stable_env.reset()
        baseline_ppo_returns.append(rewards)

        # Save results using pickle
        with open(os.path.join(results_path, "baseline_ppo_results.pkl"), "wb") as f:
            pickle.dump({"baseline_ppo_returns": baseline_ppo_returns}, f)
        print(f"Results saved to {os.path.join(results_path, 'baseline_ppo_results.pkl')}")

        tr_bar.set_description(
            f" Average Reward: {np.mean(rewards):.2f}"
        )

    env.close()
