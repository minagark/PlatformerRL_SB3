import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
import os

model_name = "DQN"
models_dir = f"models/{model_name}"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = PlatformerEnv(None) 
env.reset()

# HYPERPARAMETERS
model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
NUM_TIMESTEPS = 10_000
TIMESTEPS_MULTIPLIER = 20
for i in range(1, TIMESTEPS_MULTIPLIER+1):
    model.learn(NUM_TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
    model.save(f"{models_dir}/{NUM_TIMESTEPS * i}")

env.close()




# rewards = []
# eval_episodes = 5
# show_env = PlatformerEnv("human")
# for episode in range(eval_episodes):
#     obs, _ = show_env.reset()
#     done = False
#     while not done:
#         # show_env.render()
#         best_action, next_state = model.predict(observation=obs, deterministic=True)
#         obs, reward, done, trunc, info = show_env.step(best_action)
#         rewards.append(reward)
# show_env.close()


# plt.plot(rewards)
# plt.show()