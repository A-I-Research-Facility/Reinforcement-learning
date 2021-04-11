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
Now we need a blob class. 
'''