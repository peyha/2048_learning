import pygame
import numpy as np
import gym
from logic import *
import json
import random

window_width, window_height = 500, 500
c = json.load(open("constants.json", "r"))
screen = pygame.display.set_mode(
    (c["size"], c["size"]))
#my_font = pygame.font.SysFont(c["font"], c["font_size"], bold=True)

class GameEnv(gym.Env):
    def __init__(self,env_config={}):
        #16 cases dans la grille
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 2048,
                                                shape = (4, 4),
                                                dtype = np.uint32)

        #4 directions possibles
        self.action_space = gym.spaces.Discrete(4) 
        self.board = [[0]*4 for _ in range(4)]
        self.board = fillTwoOrFour(self.board, iter=2)
        self.nbTour = 0

    def init_render(self):
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(c["font"], c["font_size"], bold=True) 

    def reset(self):
        # reset the environment to initial state
        self.board = [[0]*4 for _ in range(4)]
        self.board = fillTwoOrFour(self.board, iter=2)
        self.nbTour = 0
        return

    def step(self, action=-1):
        # 0, 1, 2, 3 : up, down, left, right
        #print(action)
        if action == -1:
            return

        self.nbTour += 1
        directions = ["w", "s", "a", "d"]
        self.board = move(directions[int(action)], self.board)

        if not isFull(self.board):
            self.board = fillTwoOrFour(self.board)
        print(checkGameStatus(self.board))
        observation, reward, done, info = np.array(self.board), np.max(np.array(self.board)) * np.exp(-self.nbTour), checkGameStatus(self.board) != "PLAY", {}
        return observation, reward, done, info
    
    def render(self):
        theme = "light"

        self.window.fill(tuple(c["colour"][theme]["background"]))
        box = c["size"] // 4
        padding = c["padding"]
        for i in range(4):
            for j in range(4):
                colour = tuple(c["colour"][theme][str(self.board[i][j])])
                pygame.draw.rect(self.window, colour, (j * box + padding,
                                                    i * box + padding,
                                                    box - 2 * padding,
                                                    box - 2 * padding), 0)
                if self.board[i][j] != 0:
                    if self.board[i][j] in (2, 4):
                        text_colour = tuple(c["colour"][theme]["dark"])
                    else:
                        text_colour = tuple(c["colour"][theme]["light"])
                    # display the number at the centre of the tile
                    self.window.blit(self.font.render("{:>4}".format(
                        self.board[i][j]), 1, text_colour),
                        # 2.5 and 7 were obtained by trial and error
                        (j * box + 2.5 * padding, i * box + 7 * padding))
        pygame.display.update()
        
def pressed_to_action(keytouple):
    action_turn = -1
    for i, key in enumerate(keytouple):
        if key:
            if i == 81:  # back
                action_turn = 1
            elif i == 82:  # forward
                action_turn = 0
            elif i == 80:  # left  is -1
                action_turn = 2
            elif i == 79:  # right is +1
                action_turn = 3
            else:
                action_turn = -1
    # ─── KEY IDS ─────────
    # arrow forward   : 
    # arrow backwards : 
    # arrow left      : 
    # arrow right     : 
    #return np.array([action_acc, action_turn])
    return action_turn

'''
environment = GameEnv()
environment.init_render()
run = True
while run:
    # set game speed to 30 fps
    #environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    # end while-loop when window is closed
    
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            run = False
    # get pressed keys, generate action
    get_pressed = pygame.key.get_pressed()
        
    action = pressed_to_action(get_pressed)
    print(action)
    
    # calculate one step
    
    action = environment.action_space.sample()
    obs, reward, done, info = environment.step(action)
    print(done)


    # Render the game
    environment.render()

    if done == True:
        break

environment.close()
pygame.quit()

#TODO adapter cette partie à notre jeu
env = GameEnv()
env.init_render()
obs = env.reset()


while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # Render the game
    env.render()
    
    if done == True:
        break

env.close()
'''

#let's train the agent
env = GameEnv()
env.init_render()

# Traning parameters
 initial_lr = 3e-4
 discount_factor = 0.99
 gae_lambda = 0.95
 ppo_epsilon = 0.2
 value_scale = 0.5
 entropy_scale = 0.01
 horizon = 128
 num_epochs = 10
 batch_size = 128
 num_envs = 16

 # Environment constants
 num_actions = test_env.action_space.n


