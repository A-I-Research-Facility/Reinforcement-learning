'''
Introduction to deep Q-learning. 
Deep Q-learning is the combination of deep learning and reinforcement learning. Here, we create 
a RL model with a neural network similar to that of in deep learning.

This program only creates the model. We will train the model in the next program.
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np

'''
self.model is getting a .fit() every single step and that too of value 1. And we train neural networks
with a batch. 
So we are creating a batch of 50,000 steps, that we call REPLAY_SIZE_MEMORY
'''

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "256x2"

'''
By default, every time we do a .fit(), keras generates a new TensorBoard file (log file).
And we are performing that operation 200 times in 1 episode, and then, there are thousands
of episodes. But we just want one log file, that gets updated. Hence, the below class is written.
'''
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):
        '''
        The model is expected to go crazy in the beginning of the learning process. Hence, 
        we create 2 models, to ease our understanding, and deal with far less complications.
        The self.model will change in a drastic manner, the other one, not so much.
        '''
        self.model = self.create_model()        # main model, gets trained every step

        self.target_model = self.create_model()     # target model, gets predicted every step
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0      # we need this to internally track when to update target model

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape = env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))   # Rectified linear activation
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

    def get_qs(self, state, step):
        return self.model.predeict(np.array(state).reshape(-1, *state.shape) / 255)[0]      # to normalize the RGB image data that we are
                                                                                            # passing, we divide by 255
                        
