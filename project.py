import pygame
import numpy as np
import gym
from logic import *
import json

window_width, window_height = 500, 500
c = json.load(open("constants.json", "r"))
screen = pygame.display.set_mode(
    (c["size"], c["size"]))
my_font = pygame.font.SysFont(c["font"], c["font_size"], bold=True)

class GameEnv(gym.Env):
    def __init__(self,env_config={}):
        # self.observation_space = gym.spaces.Box()
        #4 directions possibles
        self.action_space = gym.spaces.Discrete(4) 



    def init_render(self):
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()

    def reset(self):
        # reset the environment to initial state
        self.board = [[0]*4 for _ in range(4)]
        self.board = fillTwoOrFour(self.board, iter=2)
        return

    def step(self, action=0):
        # 0, 1, 2, 3 : up, down, left, right
        
        observation, reward, done, info = 0., 0., False, {}
        return observation, reward, done, info
    
    def render(self):
        theme = "light"

        screen.fill(tuple(c["colour"][theme]["background"]))
        box = c["size"] // 4
        padding = c["padding"]
        for i in range(4):
            for j in range(4):
                colour = tuple(c["colour"][theme][str(self.board[i][j])])
                pygame.draw.rect(screen, colour, (j * box + padding,
                                                    i * box + padding,
                                                    box - 2 * padding,
                                                    box - 2 * padding), 0)
                if self.board[i][j] != 0:
                    if self.board[i][j] in (2, 4):
                        text_colour = tuple(c["colour"][theme]["dark"])
                    else:
                        text_colour = tuple(c["colour"][theme]["light"])
                    # display the number at the centre of the tile
                    screen.blit(my_font.render("{:>4}".format(
                        self.board[i][j]), 1, text_colour),
                        # 2.5 and 7 were obtained by trial and error
                        (j * box + 2.5 * padding, i * box + 7 * padding))
        pygame.display.update()
        
def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    if keytouple[274] == 1:  # back
        action_acc -= 1
    if keytouple[273] == 1:  # forward
        action_acc += 1
    if keytouple[276] == 1:  # left  is -1
        action_turn += 1
    if keytouple[275] == 1:  # right is +1
        action_turn -= 1
    # ─── KEY IDS ─────────
    # arrow forward   : 273
    # arrow backwards : 274
    # arrow left      : 276
    # arrow right     : 275
    return np.array([action_acc, action_turn])

environment = GameEnv()
environment.init_render()
run = True
while run:
    # set game speed to 30 fps
    environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    # end while-loop when window is closed
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            run = False
    # get pressed keys, generate action
    get_pressed = pygame.key.get_pressed()
    action = pressed_to_action(get_pressed)
    # calculate one step
    environment.step(action)
    # render current state
    environment.render()
pygame.quit()