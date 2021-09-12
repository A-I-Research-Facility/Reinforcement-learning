💢 In continuation with the previous program, we are now going to train
our DQN model.

*System requirements :-*<br>
* TensorFlow&nbsp;&nbsp;&nbsp;: 2.4.1<br>
* Dedicated GPU memory  : 8GB<br>
* Memory                : 16GB DDR4<br>
<br>

    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
    from keras.callbacks import TensorBoard
    from keras.optimizers import Adam
    from collections import deque
    import time
    import numpy as np
    from tqdm import tqdm
    import random
    import os
    from PIL import Image as Img
    import cv2

    REPLAY_MEMORY_SIZE = 50_000
    MODEL_NAME = "256x2"
    MIN_REPLAY_MEMORY_SIZE = 1_000
    
Batch size for training data :
    
    MINIBATCH_SIZE = 64
    DISCOUNT = 0.99
    UPDATE_TARGET_EVERY = 5
    MIN_REWARD = -200
    EPISODES = 20_000

    epsilon = 1
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001

Number of episodes to see stats = 100 :

    AGGREGATE_STATS_EVERY = 100
    SHOW_PREVIEW = False

***
💢 Now we want to bring in our blob class and blob environment.

    class Blob:
        def __init__(self, size):
            self.size = size
            self.x = np.random.randint(0, size)
            self.y = np.random.randint(0, size)

        def __str__(self):
            return f"Blob ({self.x}, {self.y})"

        def __sub__(self, other):
            return (self.x-other.x, self.y-other.y)

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

We have 9 total movement options : 0, 1, 2, 3, 4, 5, 6, 7, 8<br>

So, now our agent doesn't need to take help of the boundaries to go up-down or left-right :

        def action(self, choice):   
            if choice == 0:
                self.move(x=1, y=1)
            elif choice == 1:
                self.move(x=-1, y=-1)
            elif choice == 2:
                self.move(x=-1, y=1)
            elif choice == 3:
                self.move(x=1, y=-1)

            elif choice == 4:
                self.move(x=1, y=0)
            elif choice == 5:
                self.move(x=-1, y=0)

            elif choice == 6:
                self.move(x=0, y=1)
            elif choice == 7:
                self.move(x=0, y=-1)

            elif choice == 8:
                self.move(x=0, y=0)

Updated method 'move' for TF 2.4.1 :-

        def move(self, x = None, y = None):
        
If no value for x, move randomly :
            
            if x == None:
                self.x += np.random.randint(-1, 2)
            else:
                self.x += x

If no value for y, move randomly :
            
            if y == None:
                self.y += np.random.randint(-1, 2)
            else:
                self.y += y

If we are out of bounds, fix :

            if self.x < 0:
                self.x = 0
            elif self.x > self.size-1:
                self.x = self.size-1
            if self.y < 0:
                self.y = 0
            elif self.y > self.size-1:
                self.y = self.size-1

***

    class BlobEnv:
        SIZE = 10
        RETURN_IMAGES = True
        MOVE_PENALTY = 1
        ENEMY_PENALTY = 300
        FOOD_REWARD = 25
        OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
        ACTION_SPACE_SIZE = 9
        
Player key in dict :
        
        PLAYER_N = 1
        
Food key in dict :
        
        FOOD_N = 2
        
Enemy key in dict :
        
        ENEMY_N = 3
        
The dict (colours) :
        
        d = {1: (255, 175, 0),
            2: (0, 255, 0),
            3: (0, 0, 255)}

        def reset(self):
            self.player = Blob(self.SIZE)
            self.food = Blob(self.SIZE)
            while self.food == self.player:
                self.food = Blob(self.SIZE)
            self.enemy = Blob(self.SIZE)
            while self.enemy == self.player or self.enemy == self.food:
                self.enemy = Blob(self.SIZE)

            self.episode_step = 0

            if self.RETURN_IMAGES:
                observation = np.array(self.get_image())
            else:
                observation = (self.player-self.food) + (self.player-self.enemy)
            return observation

        def step(self, action):
            self.episode_step += 1
            self.player.action(action)
        
***
💢 To make the food and enemy move, we can use :<br>
self.enemy.move()<br>
self.food.move()
 

            if self.RETURN_IMAGES:
                new_observation = np.array(self.get_image())
            else:
                new_observation = (self.player-self.food) + (self.player-self.enemy)

            if self.player == self.enemy:
                reward = -self.ENEMY_PENALTY
            elif self.player == self.food:
                reward = self.FOOD_REWARD
            else:
                reward = -self.MOVE_PENALTY

            done = False
            if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
                done = True

            return new_observation, reward, done

        def render(self):
            img = self.get_image()
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            cv2.waitKey(1)

***
💢 FOR CNN :-

        def get_image(self):
        
Starts an rbg of our size :
        
            env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
            
Sets the food location tile to green color :
            
            env[self.food.x][self.food.y] = self.d[self.FOOD_N]
            
Sets the enemy location to red :
            
            env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]
            
Sets the player tile to blue :
            
            env[self.player.x][self.player.y] = self.d[self.PLAYER_N]
            
Reading to rgb, even tho color definitions are bgr :
            
            img = Img.fromarray(env, 'RGB')
            
            return img

***
💢 Initailizing environment :

    env = BlobEnv()

For stats :

    ep_rewards = [-200]

For more repetitive results :

    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

Create models folder :

    if not os.path.isdir('models'):
        os.makedirs('models')

***
💢 Following class is modified to work with TensorFlow 2.4.1

Please do not change this class in any manner, or the program may not work at all. If you face some issue regarding this class, please create a discussion.

    class ModifiedTensorBoard(TensorBoard):
    
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.step = 1
            self.writer = tf.summary.create_file_writer(self.log_dir)
            self._log_write_dir = self.log_dir

        def set_model(self, model):
            self.model = model

            self._train_dir = os.path.join(self._log_write_dir, 'train')
            self._train_step = self.model._train_counter

            self._val_dir = os.path.join(self._log_write_dir, 'validation')
            self._val_step = self.model._test_counter

            self._should_write_train_graph = False

        def on_epoch_end(self, epoch, logs=None):
            self.update_stats(**logs)

        def on_batch_end(self, batch, logs=None):
            pass

        def on_train_end(self, _):
            pass

        def update_stats(self, **stats):
            with self.writer.as_default():
                for key, value in stats.items():
                    tf.summary.scalar(key, value, step = self.step)
                    self.writer.flush()

***
💢 Foloowing class is already explained in previous codes. Please go through them if you face any issues.

    class DQNAgent:
        def __init__(self):
            self.model = self.create_model() 

            self.target_model = self.create_model() 
            self.target_model.set_weights(self.model.get_weights())

            self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

            self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
            
            self.target_update_counter = 0      

        def create_model(self):
            model = Sequential()
            model.add(Conv2D(256, (3, 3), input_shape = env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation = "linear"))
            model.compile(loss="mse", optimizer = Adam(lr=0.001), metrics=['accuracy'])

            return model
    
        def update_replay_memory(self, transition):
            self.replay_memory.append(transition)

        def get_qs(self, state):
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

        def train(self, terminal_state, step):
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                return
        
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

***
💢 By dividing in the following statement, we are trying to scale the images between 0 and 1 because that is the best way to teach convolutional neural networks :

            current_states = np.array([transition[0] for transition in minibatch]) / 255
            
            current_qs_list = self.model.predict(current_states)

***
💢 Current states after actions are taken :

            new_current_states = np.array([transition[3] for transition in minibatch]) / 255
            future_qs_list = self.target_model.predict(new_current_states)

***
💢 Following list will be the images from the game :

            X = []
            
Following list will be the actions that model decides to take :
            
            y = []

***
💢 With the following loop, we will be able to calculate the last bit (learned value) of the Q-value formula :
        
            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(current_state)
                y.append(current_qs)

            self.model.fit(np.array(X) / 255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False,
            
***
💢 Following command means that we will fit only if we are on the terminal state :
            
            callbacks = [self.tensorboard] if terminal_state else None)     
        
Following statement determines if we want to update the target_model yet :
        
            if terminal_state:
                self.target_update_counter += 1
        
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

***
💢 Create agent :

    agent = DQNAgent()

***
💢 Now we are ready to iterate over everything.

    for episode in tqdm(range(1, EPISODES + 1), ascii = True, unit = "episode"):
        agent.tensorboard.step = episode
    
        episode_reward = 0
        step = 1
        current_state = env.reset()

        done = False

        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

***
💢 Now we are going to append episode reward and then we will grab various aggregate stats. Then we will create a matplotlib chart with those values.
    
Append episode reward to a list and log stats (every given number of episodes) :

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

***
💢 Save model, but only when min reward is greater or equal a set value :

            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

***
💢 Decay epsilon :

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
