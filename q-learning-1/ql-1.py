import gym

env = gym.make("MountainCar-v0")    # this creates an environment with a mountain, and a car
env.reset()     

done = False

while not done:
    action = 2      # This environment has three actions, 0 = push car left, 1 = do nothing, 2 = push car right
    new_state, reward, done, _ = env.step(action)   

    env.render()    # rendering the GUI

env.close()

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE

# Initialising the Q-table :
import numpy as np
q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
                                                                                                    
print(q_table.shape)
