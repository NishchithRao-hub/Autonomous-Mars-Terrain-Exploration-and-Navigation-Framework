import numpy as np
import gym
import tqdm
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
from policy_network import PolicyNetwork
from ppo_agent import PPOAgent

if __name__ == "__main__":

    # Environment parameters
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)
    state_dim = np.prod(env.observation_space.shape)  # Flattened state space
    action_dim = env.action_space.n  # Discrete action space (4 actions)

    # Training parameters
    num_trials = 3
    episodes_per_trial = 100
    max_timesteps = conf["max_steps"]  # Max timesteps per episode
    update_interval = 10  # Update policy after this many timesteps

    # Storage for results
    ppo_returns = []
    ppo_actor_losses = []
    ppo_percentage_area_covered = []

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
        ppo_percentage_area_covered.append(proportion_covered)

        # Update progress bar description
        avg_reward = sum(rewards) / len(rewards)
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        tr_bar.set_description(
            f"Trial {trial + 1} | Avg Reward: {avg_reward:.2f} | Avg Actor Loss: {avg_actor_loss:.2f}"
            f" | Percentage Area Covered: {proportion_covered:.2%}")

    env.close()
