import tqdm
import pickle
from td3_agent import TD3Agent
import gym
import numpy as np
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == "__main__":
    """
    Main script to train and evaluate the TD3 agent on the Mars Explorer environment.

    This script defines the training parameters, initializes the environment and agent,
    runs multiple training trials, and plots the average performance of the agent.
    """

    # Defining the path to store plots
    plot_path = os.path.join("plot_figs")

    # Create the plot directory if it doesn't exist
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Initialize the environment
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)   # Initialize the environment
    state_dim = np.prod(env.observation_space.shape)             # Flattened state space (21x21 grid)
    action_dim =env.action_space.n                               # 4 discrete actions (up, down, left, right)

    # Training parameters
    num_trails = 3                                               # Number of trials
    episodes_per_trail = 100                                     # Episodes per training trial
    batch_size = 64                                              # Batch size for experience replay

    # Initialize storage for results
    td3_returns = []
    td3_actor_losses = []
    td3_critic_losses = []
    td3_steps = []
    td3_percentage_area_covered = []

    results_path = os.path.join("training_results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    """ 
    Train the TD3 agent for multiple trials and collect performance data.
    """
    tr_bar = tqdm.trange(num_trails)
    for trial in tr_bar:
        agent = TD3Agent(state_dim, action_dim, batch_size)
        rewards, actor_losses, critic_losses, steps, proportion_covered = agent.train(episodes=episodes_per_trail)

        td3_returns.append(rewards)
        td3_actor_losses.append(actor_losses)
        td3_critic_losses.append(critic_losses)
        td3_steps.append(steps)
        td3_percentage_area_covered.append(proportion_covered)

        tr_bar.set_description(
            f" Average Reward: {sum(rewards)/len(rewards):.2f} | Average Critic Loss: {sum(critic_losses)/len(critic_losses):.2f} | "
            f"Average Actor Loss: {sum(actor_losses)/len(actor_losses):.2f} | Percentage Area Covered: {proportion_covered:.2%}"
        )

    # Save results using pickle
    with open(os.path.join(results_path, "td3_results.pkl"), "wb") as f:
        pickle.dump({"td3_returns": td3_returns,
                     "td3_actor_losses": td3_actor_losses,
                     "td3_critic_losses": td3_critic_losses,
                     "td3_steps": td3_steps,
                     "td3_percentage_area_covered": td3_percentage_area_covered}, f)
    print(f"Results saved to {os.path.join(results_path, 'td3_results.pkl')}")

    env.close()