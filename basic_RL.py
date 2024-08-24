import gymnasium as gym
from environment import PlatformerEnv
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np

env = PlatformerEnv("human")
check_env(env)
env.reset()


episodes = 1
rewards = []
observations = []
for episode in range(episodes):
    done = False
    obs, _ = env.reset()
    while not done:
        random_action = env.action_space.sample()
        # print(f"action: {random_action}")
        obs, reward, done, trunc, info = env.step(random_action)
        rewards.append(reward)
        observations.append(obs)

# rewards = np.array(rewards)
# plt.scatter(np.arange(len(rewards)), rewards)
# plt.show()
# cumulative = np.cumsum(rewards)
# plt.scatter(np.arange(len(cumulative)), cumulative)
# plt.show()
# print(observations[:10])

env.close()