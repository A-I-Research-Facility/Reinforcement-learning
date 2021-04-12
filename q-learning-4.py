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