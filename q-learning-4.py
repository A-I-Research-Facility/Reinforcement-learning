'''
This is a tutorial to create your own Q-learning environment. We need opencv for this, so 
make sure to 'pip install' it. We also need a python imaging library, called pillow, so also
install that before starting. 

This environment is like a snakes game with a slight modification. There is a player, represented
as a blue blob, an enemy, represented as a red blob, and food, represented as a green blob.

When the environment renders, we will see all these 3 blobs on a black background. When we start
training, we will start to see results. Result analysis :
1) If red and blue blob remain on sceen, means our agent got to the food
2) If green and red blob remain, means our agent got to the enemy

Interesting thing to note : We will restrict our agent's movement to diagonals only, so it
can't move up or down in a straight line. But, it will learn on its own, to use the boundaries
of the grid (edges of the render window), even though it doesn't know about it, to correct 
its direction and go to the required place.
'''

import numpy as np
from PIL import Image as Img
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
SHOW_EVERY = 500

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
        q_table = pickle.load(f)

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
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        '''
        Now, we are done with the Q-learning. We now want to see the environment and track the metrics
        '''
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)   # this is all zeros so it is a black environment for now
            env[food.y][food.x] = d[FOOD_N]     # if we add multiple foods, the dictionary method is a big help
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            '''
            For now, we have a coloured grid, but it is still just a 10 x 10 grid.
            We want to make it an image now.
            '''
            img = Img.fromarray(env, "RGB")       # even though we are saying RGB here, it still defines as BGR. That is why
                                                    # we defined the colours in BGR format in the beginning.
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode = "valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY} mov_avg")
plt.xlabel("Episode")
plt.show()

with open(f"qt-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)