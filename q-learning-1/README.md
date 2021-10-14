💢 This is a program to open and explore a basic Q-learning environment.

We use the 'gym' library to open an environment called 'MountainCar-v0". It is
a simple environment consisting of 2 hills, a car and a destination marked with a
flag. The goal of our model will be to make the car reach the flag. The environment
only allows 3 actions for the car : <br>
a) go left, <br>
b) go right, <br>
c) don't move

This program serves as an introduction to Q-tables, how they are created, and used.

The evironment used over here doesn't make a difference to our model, but we are using
it because of its simplicity.

    import gym

Creating an environment with mountain and a car :-

    env = gym.make("MountainCar-v0")
    env.reset()     

Resetting is the first thing to do after we create an environment, then we are ready to iterate through it.

***
💢 This environment has three actions, 0 = push car left, 1 = do nothing, 2 = push car right

    done = False

    while not done:
        action = 2

        new_state, reward, done, _ = env.step(action)
        
Everytime we step through an action, we get a new_state from environment
For our sake of understanding, we know that the state returned by the 
environment is position and velocity.
Note : the states returned over her are continuous. We need to convert them
to discrete or else our model will continue to train in a never ending scenario.
We will do this conversion at the necessary time.
    
    env.render()    # rendering the GUI

    env.close()

***
💢 When we run this program, we see a car trying to climb the hill. But it isn't able to because it needs more momentum.
So, now we need to do that.

What we require, technically, is a mathematical function. But, in reality, we are just going to take the python form of it.
That python code we are creating now is called Q-table. It's a large table, that carries all possible combinations of
position and velocity of the car. We can just look at the table, to get our desired answer.

We initialise the Q-table with random values. So, first our agent explores and does random stuff, but slowly updates those
Q-values with time.

***
💢 To check all observations and all possible actions, run following (only works in gym environments):

    # print(env.observation_space.high)     # Output : [0.6     0.07]

    # print(env.observation_space.low)      # Output : [-1.2     -0.07]

    # print(env.action_space.n)             # Output : 3

***
💢 We want our Q-table to be of managable size. However, hardcoding the size is not the right move, since a real
RL model would not have this hardcoded beacuse it will change with environment.

    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 

20 * the length of any random observation space thing = [20] * 2

We are trying to separate the range of observation into 20 discrete chunks. Now we need to know the size of
those chunks.

    discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE

    # print(discrete_os_win_size)     # Output : [0.09     0.007]

***
💢 Q-values, the action with largest q-value is chosen to be performed by the agent. Initially it doesn't happen. But overtime the agent realises what to do.

|Combinations/Actions|  0  |  1  |  2  |
|--------------------|-----|-----|-----|
|        C1          |  0  |  2  |  2  |                                                         
|        C2          |  0  |  1  |  2  |                  
|        C3          |  2  |  0  |  1  |
     
***
💢 Initialising the Q-table :

    import numpy as np
    
    q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))   
    
low = lowest value, <br>
high = highest value<br>
size = 3 dimensional table, <br>
thus having a Q-value for every possible combination of actions.
    
    print(q_table.shape)
