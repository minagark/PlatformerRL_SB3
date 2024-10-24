import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# note that some of the 8 plats ones require 10 plats instead
model_name = "DQN-10Plats-5Steps-onGround-startlow-v1"
models_dir = f"models/{model_name}"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = PlatformerEnv(None) 
env.reset()

# HYPERPARAMETERS
model = DQN("MultiInputPolicy", env, verbose=0, tensorboard_log=log_dir, exploration_fraction=0.5, exploration_final_eps=0.1, buffer_size=200_000)
NUM_TIMESTEPS = 50_000
TIMESTEPS_MULTIPLIER = 20

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

env.close()
