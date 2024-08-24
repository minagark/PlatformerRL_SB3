import pygame
from game import Game

if __name__ == "__main__":
    pygame.init()
    pygame.display.init()

    WIDTH = 800
    HEIGHT = 600
    GRAVITY = 0.5

    Game1 = Game(WIDTH, HEIGHT, 60, GRAVITY, human_mode=True)

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    while Game1.running:
        clock.tick(Game1.FRAME_RATE)
        Game1.handle_input()
        Game1.update_everything()
        Game1.draw_everything(screen)
    
    pygame.quit()