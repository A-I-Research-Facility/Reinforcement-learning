import gym
import numpy as np

env = gym.make("MountainCar-v0")

# Now we are going to add certain constants. Their use will be explained later.
LEARNING_RATE = 0.1
DISCOUNT = 0.95     # measure of how much we value future reward over current reward (>0, <1)
EPISODES = 250

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

# We need to convert the continuous states to discrete states. For that, we need a helper function
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # We need this returned in tuple form. Hence, 
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())

# print(discrete_state)       # Output : (7,  10) {could be anything}

# We can now lookup that discrete state in the Q-table, and find the maximum Q-value
# print(np.argmax(q_table[discrete_state]))

# Since now we are ready with our new discrete state, our model can take action, and start generating
# new Q-table.
# We now require the while loop from previous program, but instead of hardcoded values, we will use dynamic values
done = False

while not done:
    action = np.argmax(q_table[discrete_state])     # we will have new discrete state soon
    new_state, reward, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    env.render()
    
    # The environment might be over already, but if it is not
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])      # we use np.max() instead of argmax() beacuse we will use 
                                                                # max_future_q in our new Q formula, so we want the Q-value
                                                                # instead of the argument.
                                                                # Slowly overtime, Q-value gets back propagated down the table

        # Finding the current Q-value
        current_q = q_table[discrete_state + (action, )]

        # The new Q-formula
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)    # The way Q-value back propagates is based
                                                                                                        # on all the parameters of this formula
        q_table[discrete_state + (action, )] = new_q        # updating the Q-table based on the newest Q-value

    elif new_state[0] >= env.goal_position:
        q_table[discrete_state + (action, )] = 0

    discrete_state = new_discrete_state

env.close()