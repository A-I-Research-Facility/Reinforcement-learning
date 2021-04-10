# The evironment used over here doesn't make a difference to our model, but it's easy to understand

import gym

env = gym.make("MountainCar-v0")    # this creates an environment with a mountain, and a car
env.reset()     # Resetting is the first thing to do after we creaate an environment
                # Then we are ready to iterate through it
done = False

while not done:
    action = 2      # This environment has three actions, 0 = push car left, 1 = do nothing, 2 = push car right
    new_state, reward, done, _ = env.step(action)   # everytime we step through an action, we get a new_state from environment
                                                    # For our sake of understanding, we know that the state returned by the 
                                                    # environment is position and velocity.
    env.render()    # rendering the GUI

env.close()

# When we run this program, we see a car trying to climb the hill. But it isn't able to because it needs more momentum.
# So, now we need to do that

# What we require, technically, is a mathematical function. But, in reality, we are just going to take the python form of it.
# That python code we are creating now is called Q-table. It's a large table, that carries all possible combinations of
# position and velocity of the car. We can just look at the table, to get our desired answer.

# We initialise the Q-table with random values. So, first our agent explores and does random stuff, but slowly updates those
# Q-values with time.

# 