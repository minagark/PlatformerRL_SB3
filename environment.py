import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Game
import pygame

class PlatformerEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 24}

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,2), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "agent_x": spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),
                "agent_vel": spaces.Box(low=-50, high=50, shape=(2,), dtype=np.float32),
                "on_ground": spaces.Discrete(n=2, start=0),
                "platform_info": spaces.Box(low=-500, high=500, shape=(20,), dtype=np.float32) 
            }
        )

        self.episode_ended = False
        self.WIDTH = 800
        self.HEIGHT = 600

        self.Game = Game(self.WIDTH, self.HEIGHT, 60, 0.5, False)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.episode_ended = False
        self.Game = Game(800, 600, 60, 0.5, False)

        observation, reward, done = self.Game.ML_step_nx(0, 1)

        if self.render_mode == "human":
            self._render_frame()

        return observation, dict()

    def step(self, action):
        if action in [0,1,2,3]:
            observation, reward, terminated = self.Game.ML_step_nx(action, 5)
        else:
            raise ValueError("Action can only be 0, 1, 2, or 3 (nothing, move left, move right, or jump).")

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, dict()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self.Game.get_canvas()
        
        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()

            self.clock.tick(self.metadata["render_fps"])

    # def render(self):
    #     if self.render_mode == "human":
    #         clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
