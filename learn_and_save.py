import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os

model_name = "DQN-8Plats-5Steps-onGround-v1"
models_dir = f"models/{model_name}"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = PlatformerEnv(None) 
env.reset()

# HYPERPARAMETERS
model = DQN("MultiInputPolicy", env, verbose=0, tensorboard_log=log_dir, exploration_fraction=0.5, buffer_size=50_000)
NUM_TIMESTEPS = 5_000
TIMESTEPS_MULTIPLIER = 40

checkpoint_callback = CheckpointCallback(
    save_freq=NUM_TIMESTEPS,
    save_path=f"{models_dir}",
    name_prefix=model_name,
    save_replay_buffer=True
)

model.learn(
    total_timesteps=NUM_TIMESTEPS*TIMESTEPS_MULTIPLIER, 
    reset_num_timesteps=False, 
    tb_log_name=model_name, 
    callback=checkpoint_callback,
)

# for i in range(1, TIMESTEPS_MULTIPLIER+1):
#     model.learn(NUM_TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
#     model.save(f"{models_dir}/{NUM_TIMESTEPS * i}")

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