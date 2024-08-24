import gymnasium as gym
from environment import PlatformerEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
import os

env = PlatformerEnv(None) 
env.reset()

model_name = "DQN"
models_dir = f"models/{model_name}"
model_path = f"{models_dir}/130000.zip"

model = DQN.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        best_action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(best_action)