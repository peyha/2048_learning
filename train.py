import random
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple
import json
import pygame

from logic import *


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
        self.lastAction = -1
        self.nbRep = 0
        self.lastReward = 0

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
        self.lastReward = 0
        return np.array(self.board).flatten()

    def step(self, action=-1):
        # 0, 1, 2, 3 : up, down, left, right
        #print(action)
        if action == -1:
            return
        LIM = 50
        #print(action, self.nbTour)
        self.nbTour += 1
        directions = ["w", "s", "a", "d"]
        last_max = np.max(np.array(self.board))
        old_board = np.array(self.board).copy()


        
   
        self.board = move(directions[int(action)], self.board)

        if np.sum(old_board == np.array(self.board)) == 16:
            self.nbRep += 1
        else:
            self.nbRep = 0
        
        if self.nbRep > LIM:
            action = random.randint(0, 3)
            self.board = move(directions[int(action)], self.board)  

        cur_max = np.max(np.array(self.board))
        reward = (cur_max - self.lastReward) / 2048
        tmp = self.lastReward
        self.lastReward= (cur_max - self.lastReward) / 2048
 
        if not isFull(self.board):
            self.board = fillTwoOrFour(self.board)

        observation, reward, done, info = np.array(self.board).flatten(), reward, checkGameStatus(self.board) != "PLAY", {}
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

class DQN(nn.Module):
    ## neural network for the rl agent

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                nn.Linear(4*4,16*16),
                nn.ReLU(),
                nn.Linear(16*16,16*6),
                nn.ReLU(),
                nn.Linear(16*6, 16),
                nn.ReLU(),
                nn.Linear(16,4)
                )

    def forward(self, x):
        #print(x, self.linear(x))
        #x = x.flatten().to(device)
        return self.linear(x.to(device))

class ReplayMemory(object):
    #memory of the training
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) 
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory) 

def choose_action_epsilon_greedy(net, state, epsilon):
    
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)
    
    temperature = 4. # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
    
    # Get the best action (argmax of the network output)
    all_possible_actions = []
    best_action = -1
    for i, b in enumerate(legal_moves(env.board)):
        if b and (best_action == -1 or net_out[i] > net_out[best_action]):
            best_action = i

    #best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions (this list includes all the actions but the optimal one)
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly from non_optimal_actions
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.cpu().numpy()

def choose_action_softmax(net, state, temperature):
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature < 0.:
        return choose_action_epsilon_greedy(net, state, 0)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32)
        net_out = net(state)

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
    
    
    # Sample the action using softmax output as mass pdf
    all_possible_actions = []
    for i, b in enumerate(legal_moves(env.board)):
        if b:
            all_possible_actions.append(i)
    net_out = net_out[all_possible_actions]
    softmax_out = nn.functional.softmax(net_out/temperature, dim=0).cpu().numpy()
    softmax_out /= np.sum(softmax_out)
    #if naan, we take randomly
    if np.any(np.isnan(softmax_out)):
        print("overflow detected")
        print(all_possible_actions, softmax_out, net_out)
        softmax_out = np.ones(len(all_possible_actions))
        softmax_out = softmax_out / np.sum(softmax_out) 
    # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    action = np.random.choice(all_possible_actions, p=softmax_out)
    
    return action, net_out.cpu().numpy()

# Real Training 
env = GameEnv() 
env.seed(0) # Set a random seed for the environment 

state_space_dim = env.observation_space.shape
action_space_dim = env.action_space.n
device = torch.device("cpu")

print(f"STATE SPACE SIZE: {state_space_dim}")
print(f"ACTION SPACE SIZE: {action_space_dim}")

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

gamma = 0.99  
replay_memory_capacity = 10000   
lr = 1e-6
target_net_update_steps = 10   
batch_size = 64
bad_state_penalty = -20000
min_samples_for_training = 1000

# replay memory
print("creating de replay memory")
replay_mem = ReplayMemory(replay_memory_capacity)    

# policy network
print("creating the policy network")
policy_net = DQN(state_space_dim, action_space_dim).to(device)

# target network with the same weights of the policy network
print("creating the target network")
target_net = DQN(state_space_dim, action_space_dim).to(device)
target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

loss_fn = nn.MSELoss()

def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # Sample from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device=device) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute Q values 
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device=device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)# Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping 
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

### Define exploration profile (decreasing exponential)
initial_value = 5.
num_iterations = 2000
exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) 
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]
epsilon_profile = 1 - np.linspace(0., 0.3, num_iterations // 2)

epsilon_profile = np.concatenate([epsilon_profile, np.zeros(num_iterations - num_iterations // 2)])

afficher = False


env = GameEnv()
if afficher:
    window_width, window_height = 500, 500
    c = json.load(open("constants.json", "r"))
    screen = pygame.display.set_mode(
        (c["size"], c["size"]))
    env.init_render()
env.seed(0)


#print(legal_moves([[2,2,8,4],[4,16,4,2],[2,4,2,4],[4,2,4,2]]))
plotting_rewards=[]
moyennes = []
moy = 0
for episode_num, eps in enumerate(exploration_profile):

    print("Achieved a score of {}".format(np.max(np.array(env.board))))
    state = env.reset()
    score = 0
    done = False

    while not done:

      # Choose the action following the policy
      #action, q_values = choose_action_softmax(policy_net, state, temperature=tau)
      action, q_values = choose_action_softmax(policy_net, state, eps)
      next_state, reward, done, info = env.step(action)

      # Update the final score (-1 for each step)
      score += reward

      if done:  
          reward += bad_state_penalty
          next_state = None
      
      # Update the replay memory
      replay_mem.push(state, action, next_state, reward)

      # Update the network
      if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
          update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

      # Visually render the environment 
      if afficher:
        env.render()

      # Set the current state for the next iteration
      state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
    plotting_rewards.append(score)
    moy += score
    if episode_num % 10 == 9:
        moyennes.append(moy / 10)
        moy = 0
        print(f"Moyenne des 10 derniers tests at iteration {episode_num}: {moyennes[-1]}")


    #print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Epsilon: {eps}") # Print the final score

env.close()
plt.plot(moyennes)
plt.show()