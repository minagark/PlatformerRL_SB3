import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
import os

env = PlatformerEnv(render_mode="human") 
env.reset()

model_name = "DQN-16Plats-5Steps-v1"
models_dir = f"models/{model_name}"
model_path = f"{models_dir}/{model_name}_150000_steps"

model = DQN.load(model_path, env=env)

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        best_action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(best_action)
