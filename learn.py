import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN

env = PlatformerEnv(None)
env.reset()

# HYPERPARAMETERS
NUM_TIMESTEPS = 100_000



model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(NUM_TIMESTEPS)
env.close()

rewards = []
eval_episodes = 5
show_env = PlatformerEnv("human")
for episode in range(eval_episodes):
    obs, _ = show_env.reset()
    done = False
    while not done:
        show_env.render()
        best_action, next_state = model.predict(observation=obs, deterministic=True)
        obs, reward, done, trunc, info = show_env.step(best_action)
        rewards.append(reward)

show_env.close()
# plt.plot(rewards)
# plt.show()