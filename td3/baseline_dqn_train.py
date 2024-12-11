from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
import tqdm
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":
    """
    Main script to train and evaluate the DQN agent on the Mars Explorer environment.

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
    num_trails = 10  # Number of trials
    episodes_per_trail = 5000  # Episodes per training trial

    # Initialize storage for results
    baseline_dqn_returns = []

    """
        Train Stable Baselines3 DQN Agent
    """
    # Vectorize the environment for Stable Baselines3
    stable_env = make_vec_env(lambda: gym.make('mars_explorer:exploConf-v1', conf=conf), n_envs=1)

    # Set up Stable Baselines3 DQN
    stable_dqn_agent = DQN(
        "MlpPolicy",  # Use a Multi-Layer Perceptron policy
        stable_env,
        verbose=1,
        learning_rate=1e-2,
        buffer_size=50000,  # Size of the replay buffer
        batch_size=64,  # Batch size for sampling from the replay buffer
        exploration_fraction=0.1,  # Fraction of training steps for epsilon-greedy exploration
        exploration_final_eps=0.02,  # Final value of epsilon for exploration
        target_update_interval=10,  # Frequency of target network updates
        train_freq=(1, "step"),  # Train the model every step
        gamma=0.99  # Discount factor
    )

    """ 
    Train the DQN agent for multiple trials and collect performance data.
    """
    tr_bar = tqdm.trange(num_trails, desc="Training Stable Baselines3 DQN")
    max_steps = conf.get('max_steps', 800)  # Use a default if not set

    for trial in tr_bar:
        stable_dqn_agent.learn(total_timesteps=episodes_per_trail)

        rewards = []
        obs = stable_env.reset()
        for _ in range(episodes_per_trail):
            action, _ = stable_dqn_agent.predict(obs, deterministic=True)
            obs, reward, done, info = stable_env.step(action)
            rewards.append(reward)

            if done:
                obs = stable_env.reset()
        baseline_dqn_returns.append(rewards)

        # Save results using pickle
        with open(os.path.join(results_path, "baseline_dqn_results.pkl"), "wb") as f:
            pickle.dump({"baseline_dqn_returns": baseline_dqn_returns}, f)
        print(f"Results saved to {os.path.join(results_path, 'baseline_dqn_results.pkl')}")

        tr_bar.set_description(
            f" Average Reward: {np.mean(rewards):.2f}"

        )

    env.close()
