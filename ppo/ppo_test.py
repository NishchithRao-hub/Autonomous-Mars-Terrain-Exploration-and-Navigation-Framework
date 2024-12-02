import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf
import numpy as np


# Create a function to instantiate the environment
def make_env():
    # # Create the environment instance with a custom configuration
    # conf = DEFAULT_CONFIG
    # env = ExplorerConf(conf=conf)
    env = gym.make('mars_explorer:exploConf-v1', conf=conf)
    return env

# Wrapping the environment in a vectorized environment (for PPO compatibility)
env = DummyVecEnv([make_env])

# Initialize PPO model (it uses a multi-layer perceptron policy by default)
model = PPO("MlpPolicy", env, verbose=1)

# Training the model
model.learn(total_timesteps=10)

# # Save the model
# model.save("ppo_mars_explorer")

# To test the trained model, we can run it in the environment
env = make_env()  # Make a fresh environment instance
obs = env.reset()

for i in range(100):  # Run for a few steps
    print("running:",i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("rewards:",rewards)
    env.render()  # Optionally render the environment

