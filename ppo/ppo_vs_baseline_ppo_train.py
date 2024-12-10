from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
import tqdm
import pickle
from policy_network import PolicyNetwork
from ppo_agent import PPOAgent

if __name__ == "__main__":

    results_path = os.path.join("training_results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Initialize the environment
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)  # Initialize the environment
    state_dim = np.prod(env.observation_space.shape)  # Flattened state space (21x21 grid)
    action_dim = env.action_space.n  # 4 discrete actions (up, down, left, right)

    # Training parameters
    num_trials = 5  # Number of trials
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
        verbose=0,
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
    print("------Starting training for Baseline PPO------")
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
        with open(os.path.join(results_path, "baseline_ppo_results_for_comparison.pkl"), "wb") as f:
            pickle.dump({"baseline_ppo_returns": baseline_ppo_returns}, f)
        print(f"Results saved to {os.path.join(results_path, 'baseline_ppo_results_for_comparison.pkl')}")

        tr_bar.set_description(
            f" Average Reward: {np.mean(rewards):.2f}"
        )

    env.close()

    # ------------------------------------------------------------------------------------------------------------------

    max_timesteps = conf["max_steps"]
    update_interval = 10

    # Storage for results
    ppo_returns = []
    ppo_actor_losses = []
    ppo_steps = []
    ppo_percentage_area_covered = []

    print("------Starting training for PPO------")
    # Train the PPO agent for multiple trials
    tr_bar = tqdm.trange(num_trials, desc="Training Trials")
    for trial in tr_bar:
        # Initialize the policy network and PPO agent
        policy_network = PolicyNetwork(input_size=state_dim, action_space=action_dim)
        agent = PPOAgent(policy_network, state_dim, action_dim, lr=3e-4)

        # Train the agent and collect results
        rewards, actor_losses, steps = agent.train(
            env, num_episodes=episodes_per_trial, max_timesteps=max_timesteps, update_interval=update_interval
        )

        # covered_area = env.get_covered_area()
        proportion_covered = env.get_covered_proportion()

        ppo_returns.append(rewards)
        ppo_actor_losses.append(actor_losses)
        ppo_steps.append(steps)
        ppo_percentage_area_covered.append(proportion_covered)

        # Update progress bar description
        avg_reward = sum(rewards) / len(rewards)
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        tr_bar.set_description(
            f"Trial {trial + 1} | Avg Reward: {avg_reward:.2f} | Avg Actor Loss: {avg_actor_loss:.2f}"
            f" | Percentage Area Covered: {proportion_covered:.2%}")

    # Save results using pickle
    with open(os.path.join(results_path, "ppo_results_for_comparison.pkl"), "wb") as f:
        pickle.dump({
            "ppo_returns": ppo_returns,
            "ppo_actor_losses": ppo_actor_losses,
            "ppo_steps": ppo_steps,
            "ppo_percentage_area_covered": ppo_percentage_area_covered}, f)
    print(f"Results saved to {os.path.join(results_path, 'ppo_results_for_comparison.pkl')}")

    env.close()
