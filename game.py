import pygame
import random
from entities import Entity, Platform, Player
import math
import numpy as np

class Game():
    def __init__(self, WIDTH, HEIGHT, FRAME_RATE, gravity, human_mode) -> None:
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.platforms = []
        self.player = Player(self.WIDTH/2, self.HEIGHT - 105, 15, 15, (0, 0, 255), gravity, 8)
        self.FRAME_RATE = FRAME_RATE
        self.y_level = 0
        self.old_y_level = 0
        self.running = True
        self.human_mode = human_mode
        self.step_number = 0
        self.considered_platforms = []

        self.populate_platforms()
        self.platforms.append(Platform(0, 600, 800, 20, (255, 0, 0), "lava"))

    def ML_step(self, action):
        """
        Input: action {0,1,2,3}
        Output: observation; shape (11,2) consisting of 
        position, velocity, acceleration of player and relative positions of 8 closest platforms.   
        """
        self.step_number += 1
        if action == 0:
            self.player.pressing_down = False
            self.player.pressing_right = False
            self.player.pressing_left = False
            self.player.pressing_up = False
        elif action == 1:
            self.player.pressing_down = False
            self.player.pressing_right = False
            self.player.pressing_left = True # True
            self.player.pressing_up = False
        elif action == 2:
            self.player.pressing_down = False
            self.player.pressing_right = False
            self.player.pressing_left = False
            self.player.pressing_up = True # True
        elif action == 3:
            self.player.pressing_down = False
            self.player.pressing_right = True # True
            self.player.pressing_left = False
            self.player.pressing_up = False
        
        if self.player.on_ground:
            self.old_y_level = self.y_level

        self.update_everything()

        # optimize this?
        def dist_from_here(platform):
            return math.sqrt((platform.x - self.player.x) ** 2 + (platform.y - self.player.y) ** 2)

        closest_platforms = sorted(self.platforms[:-1], key = dist_from_here)
        self.considered_platforms = closest_platforms[:10]


        ### The following doesn't work because it gives a variable-sized observation;
        ### but there are some tricks to make it work 
        # self.considered_platforms = []
        # for plat in self.platforms[:-1]:
        #     if dist_from_here(plat) < 50:
        #         self.considered_platforms.append(plat)
        # self.considered_platforms = sorted(self.considered_platforms, key=dist_from_here)
        

        agent_x = np.array([self.player.x,], dtype=np.float32)
        agent_vel = np.array([self.player.vx, self.player.vy], dtype=np.float32)
        on_ground = int(self.player.on_ground)
        platform_info = np.array([(platform.x - self.player.x, platform.y - self.player.y) 
                                    for platform in self.considered_platforms], dtype=np.float32).reshape((-1,))
        observation = {"agent_x": agent_x, 
                       "agent_vel": agent_vel, 
                       "on_ground": on_ground, 
                       "platform_info": platform_info}
        
        done = not self.running

        if self.player.on_ground or done:
            # we want negative y_level since going up is negative y
            reward = -(self.y_level - self.old_y_level)
        else:
            reward = 0.0

        return observation, reward, done

    def ML_step_nx(self, action, n):
        full_reward = 0
        for _ in range(n):
            observation, reward, done = self.ML_step(action)
            full_reward += reward
            if done:
                break
        return observation, full_reward, done

    def update_everything(self):
        self.y_level = self.player.y - 300
        self.platforms[-1].height += 0.2
        self.platforms[-1].y -= 0.2
        self.update_player()

    def get_canvas(self):
        canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        canvas.fill((100,100,100))
        for platform in self.platforms:
            self.draw_entity(platform, canvas)
            if not self.human_mode and platform in self.considered_platforms:
                pygame.draw.rect(canvas, (0,0,0), (platform.x, platform.y - self.y_level, platform.width, platform.height))
        self.draw_entity(self.player, canvas)
        return canvas

    def draw_everything(self, screen):
        pygame.draw.rect(screen, (100,100,100), (0, 0, self.WIDTH, self.HEIGHT))
        for platform in self.platforms:
            self.draw_entity(platform, screen)
            if not self.human_mode and platform in self.considered_platforms:
                pygame.draw.rect(screen, (0,0,0), (platform.x, platform.y - self.y_level, platform.width, platform.height))
        self.draw_entity(self.player, screen)
        pygame.display.flip()

    def update_player(self):
        p = self.player
        p.ax = 0

        p.vy += p.gravity

        # Stop out of bounds
        if (p.x + p.vx > self.WIDTH - p.width):
            p.x = self.WIDTH - p.width
            p.vx = 0
        if (p.x + p.vx < 0):
            p.x = 0
            p.vx = 0
        

        #Checking if the player is on the ground, and stopping its movement
        if (p.y + p.vy > self.HEIGHT - p.height):
            p.y = self.HEIGHT - p.height
            p.vy = 0
        

        if (p.pressing_left):
            p.ax = -2 
        if (p.pressing_right):
            p.ax = 2
        if (p.pressing_right and p.pressing_left):
            p.ax = 0
        
        # Update and clamp velocity
        p.vx += p.ax
        if (p.vx > p.max_vx):
            p.vx = p.max_vx
        if (p.vx < -p.max_vx):
            p.vx = -p.max_vx
        if (p.vy > p.max_vy):
            p.vy = p.max_vy
        if (p.vy < -p.max_vy):
            p.vy = -p.max_vy

        # Move player
        p.x += p.vx * 60 / self.FRAME_RATE
        p.y += p.vy * 60 / self.FRAME_RATE

        p.on_ground = False
        if ( (p.y >= self.HEIGHT - p.height)  ):
            p.on_ground = True
            p.y = self.HEIGHT - p.height
            if (p.pressing_up):
                p.vy = -p.jump_height
                # print("JUMP")
            else:
                p.vy = min(0, p.vy)
            
        
        for platform in self.platforms:
            if (platform.platform_type == "lava" and p.collision_type(platform)):
                # print(f"hit the lava....; y={p.y}")
                self.lose()

            if (p.collision_type(platform) == "bottom"):
                # if (platform.platform_type == "lava"):
                #     print(f"hit the lava....; y={p.y}")
                #     self.lose()

                p.on_ground = True
                if (p.vy > 0):
                    p.y = platform.y - p.height
                
                if (platform.platform_type == "bouncy"):
                    # print("BOING")
                    p.vy = -p.jump_height * 3/2 / math.sqrt(60 / self.FRAME_RATE)
                elif (platform.platform_type == "sticky" and p.pressing_up and p.vy > 0.01):
                    p.vy = -p.jump_height / 3 * 2 / math.sqrt(60 / self.FRAME_RATE)
                    # print("smol jump")
                elif (p.pressing_up and p.vy > 0.01):
                    p.vy = -p.jump_height / math.sqrt(60 / self.FRAME_RATE)
                    # print("JUMP")
                else:
                    p.vy = min(0, p.vy)
                
            
            if (p.collision_type(platform) == "top" and platform.platform_type == "hard"):
                p.y = platform.y + platform.height
                p.vy = 0
                # print("BONK")
            
        
    
        #friction-ish: if 0 outside acceleration, slow down the player
        if (p.ax == 0):
            if (p.vx > 0):
                p.vx = max(0, p.vx - 0.5)
            
            if (p.vx < 0):
                p.vx = min(0, p.vx + 0.5)
            
    def draw_entity(self, e: Entity, screen):
        pygame.draw.rect(screen, e.color, (e.x, e.y - self.y_level, e.width, e.height))

    def populate_platforms(self):
        #platform_options = [["green", "normal"], ["blue", "bouncy"], ["purple", "sticky"], ["gray", "hard"]]
        self.platforms.append(Platform(self.WIDTH/2, self.HEIGHT - 100, 50, 5, (0,255,0), "normal"))
        for y_offset in range(0, self.HEIGHT * 10, self.HEIGHT):
            # Human game version will have tapering platforms higher up, and different types of platforms. 

            vertical_density = 12
            horizontal_density = 6

            vertical_interval = self.HEIGHT // vertical_density
            horizontal_interval = self.WIDTH // horizontal_density

            for x in range(0, self.WIDTH, horizontal_interval):
                for y in range(0, self.HEIGHT, vertical_interval):
                    color = (0,255,0)
                    platform_type = "normal"
                    self.create_and_add_random_platform(x, y - y_offset, horizontal_interval, vertical_interval, 50, 5, color, platform_type)

    def create_and_add_random_platform(self, start_x, start_y, x_interval, y_interval, width, height, color, platform_type):
        x = random.random() * x_interval + start_x
        y = random.random() * y_interval + start_y
        self.platforms.append(Platform(x, y, width, height, color, platform_type))

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    self.player.pressing_right = True
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    self.player.pressing_down = True
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    self.player.pressing_left = True
                elif event.key in (pygame.K_w, pygame.K_UP, pygame.K_SPACE):
                    self.player.pressing_up = True

            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    self.player.pressing_right = False
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    self.player.pressing_down = False
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    self.player.pressing_left = False
                elif event.key in (pygame.K_w, pygame.K_UP, pygame.K_SPACE):
                    self.player.pressing_up = False
            
            if event.type == pygame.QUIT:
                self.lose()

    def lose(self):
        self.running = False