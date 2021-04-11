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

start_q_table = None    # We can add an existing Q-table by writing the full path here

