import gym
import numpy as np
import matplotlib.pyplot as plt
from td3_agent import TD3Agent
from utils import plot_trajectory
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

def get_conf():
    conf["size"] = [30, 30]
    conf["obstacles"] = 20
    conf["lidar_range"] = 4
    conf["obstacle_size"] = [1, 3]

    conf["viewer"]["night_color"] = (0, 0, 0)
    conf["viewer"]["draw_lidar"] = True

    conf["viewer"]["drone_img"] = "mars-explorer/tests/img/drone.png"
    conf["viewer"]["obstacle_img"] = "mars-explorer/tests/img/block.png"
    conf["viewer"]["background_img"] = "mars-explorer/tests/img/mars.jpg"
    conf["viewer"]["light_mask"] = "mars-explorer/tests/img/light_350_hard.png"
    return conf

if __name__ == "__main__":
    conf = get_conf()
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)

    # Initialize TD3 agent
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    max_action = 1
    agent = TD3Agent(state_dim, action_dim, max_action)

    episodes = 10
    max_steps = 1000
    total_rewards = []
    terrain_coverage = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        visited = set()

        for step in range(max_steps):
            env.render()
            # Get action from TD3 agent
            action = agent.select_action(np.array(state))

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            # Log terrain coverage
            x, y = info.get('position', (0, 0))
            visited.add((int(x), int(y)))

            # Add to the replay buffer and update agent
            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            if done:
                break

        total_rewards.append(ep_reward)
        terrain_coverage.append(len(visited) / (conf["size"][0] * conf["size"][1]))

        print(f"Episode {ep + 1}: Reward = {ep_reward}, Terrain Coverage = {terrain_coverage[-1]:.2%}")

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards, label="Total Reward")
    plt.plot([tc * 100 for tc in terrain_coverage], label="Terrain Coverage (%)")
    plt.xlabel("Episodes")
    plt.ylabel("Value")
    plt.title("TD3 Performance")
    plt.legend()
    plt.show()

    env.close()
