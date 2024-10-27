import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os

env = PlatformerEnv() 
env.reset()

log_dir = "logs"
model_name = "DQN-10Plats-5Steps-onGround-startlow-v1"
models_dir = f"models/{model_name}"
model_path = f"{models_dir}/{model_name}_900000_steps"

model = DQN("MultiInputPolicy", env, verbose=0, tensorboard_log=log_dir, exploration_fraction=0.75, exploration_initial_eps=0.3, exploration_final_eps=0.1, buffer_size=200_000)
model.set_parameters(model_path)

# model = DQN.load(model_path, env=env)

model.load_replay_buffer(f"{models_dir}/{model_name}_replay_buffer_900000_steps")

model_name = "DQN-10Plats-5Steps-onGround-startlow-v1.5"
models_dir = f"models/{model_name}"

NUM_TIMESTEPS = 50_000
TIMESTEPS_MULTIPLIER = 20

checkpoint_callback = CheckpointCallback(
    save_freq=NUM_TIMESTEPS,
    save_path=models_dir,
    name_prefix=model_name,
    save_replay_buffer=True
)

model.learn(
    total_timesteps=NUM_TIMESTEPS*TIMESTEPS_MULTIPLIER,
    reset_num_timesteps=True,
    tb_log_name=model_name,
    callback=checkpoint_callback
)