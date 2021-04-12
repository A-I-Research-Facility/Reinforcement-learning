'''
This is a tutorial to create your own Q-learning environment. We need opencv for this, so 
make sure to 'pip install it'. We also need a python imaging library, called pillow, so also
install that before starting. 
'''

import numpy as np
import PIL as Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

# Constants needed for our program
SIZE = 10       # We are making a 10 x 10 grid
HM_EPISODES = 25000     # how many episodes
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

start_q_table = None    # We can add an existing Q-table by writing the full path here

# Defining colour for our objects. This is in BGR format
d = {1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)}

'''
Now we need a blob class. All the blobs have a lot of same attributes, like, movement directions,
starting locations, etc.

We are defining observation in our environment as the relative location of food, and the relative
location of the enemy. So, we need to create a blob class to handle all of the repeated things with ease.
'''

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        '''
        Few things to note here that we want to avoid:
        1) The player could spawn on the food;
        2) Player could spawn on enemy;
        3) Enemy could spawn on the food.

        We want to avoid all the above scenarios. But since we are making a very simple
        environment here, we won't worry about that for now.
        '''

    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):   # We are making this method to interact with another method in case we want to add more players
        '''
        The moves that we are coding here only allows diagonal movement. To make a blob go
        up-down and side to side, add the codes for only x, and only y.
        '''
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = 1, y = -1)

    def move(self, x = False, y = False):
        '''
        We wanna be able to move randomly if not value is passed,
        or very specifically otherwise.
        '''
        if not x:
            self.x += np.random.randint(-1, 2)  # this can attain value 0, so this random movement can be up down
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)  # this can attain value 0, so this random movement can be up down
        else:
            self.y += y
        
        '''
        Remember that we are making a 10 x 10 grid. We need to create movement boundaries for
        our agent.
        '''
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

'''
We are finished with our blob class. Now, we either want to create a Q-table, or load an existing one.
'''
if start_q_table is None:
    q_table = {}
    '''
    Our observation space will look like :
    (x1, y1), (x2, y2)
    So, to iterate through every possibility, we need 4 nested FOR loops.
    '''
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickel.load(f)

'''
Now, we have our training to do.
'''
episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon : {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0

    for i in range(200):
        obs = (player - food, player - enemy)   # Operator overloading
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        
        player.action(action)
        '''
        Later, we might wanna make the enemy and the food move. For that, use the commands below :
        enemy.move()
        food.move()
        But for training purposes now, it is better to not let them move initially in order to keep
        things simple.
        
        Now, assigning the rewards to actions
        '''
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        # To make our Q function, we need to be able to make a new observation based on the movement
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        # Now we are ready to calculate our Q-function
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * cuurent_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        '''
        Now, we are done with the Q-leaarning. We now want to see the environment and track the metrics
        '''
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)   # this is all zeros so it is a black environment for now
            env[food.x][food.y]