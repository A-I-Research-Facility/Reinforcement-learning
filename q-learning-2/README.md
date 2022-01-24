💢 Progressing on what we developed in the previous program, this one deals with the 
agent actually achieving its goal. 

We create multiple episodes for our agent to learn. Also, use of epsilon and its functions
are introduced here.

    import gym
    import numpy as np

    env = gym.make("MountainCar-v0")

***
💢 Now we are going to add certain constants. Their use will be explained later.
    
    LEARNING_RATE = 0.1

Measure of how much we value future reward over current reward (> 0, < 1) :-

    DISCOUNT = 0.95    
    EPISODES = 25000

    SHOW_EVERY = 2000

    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OS_SIZE

***
💢 Some models require some random actions to be taken to ge the desired result. For this, we need to define
`EPSILON` over here. Even though in this case, our model is able to achive the goal without requiring
this varible. Also, the value of epsilon varies between 0 and 1 only.

Epsilon basically helps the model explore in random directions. It is suprising what the model finds
out sometimes. The higher the epsilon, the more likely the model is to perform a random action.

    epsilon = 0.5
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2

    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING) # Amount decayed by in each episode

    q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

***
💢 We need to convert the continuous states to discrete states. For that, we need a helper function.

    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        
   We need this returned in tuple form. Hence, 
   
        return tuple(discrete_state.astype(np.int))

***
💢 Now we want to iterate over episodes. Since currently the model only runs one time
and we want more than that.

    for episode in range(EPISODES):
        if episode % SHOW_EVERY == 0:
            print(episode)
            render = True
        else:
            render = False

        discrete_state = get_discrete_state(env.reset())

        # print(discrete_state)       # Output : (7,  10) {could be anything}


   We can now lookup that discrete state in the Q-table, and find the maximum Q-value.
    
    # print(np.argmax(q_table[discrete_state]))

   ***
   💢 Since now we are ready with our new discrete state, our model can take action, and start generating
   new Q-table.
   <br>
   
   We now require the while loop from previous program, but instead of hardcoded values, we will use dynamic values.

        done = False

        while not done:
            if np.random.random() > epsilon:
        
   We will have new discrete state soon :
   
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
                
            
            new_state, reward, done, _ = env.step(action)
            
            new_discrete_state = get_discrete_state(new_state)
        
            if render:
                env.render()
        
   The environment might be over already, but if it is not, we use the following command. We use np.max() instead of argmax() beacuse we will use
   max_future_q in our new Q formula, so we want the Q-value instead of the argument. Slowly overtime, Q-value gets back propagated down the table.
   
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])

   Finding the current Q-value :
   
                current_q = q_table[discrete_state + (action, )]

   The new Q-formula (The way Q-value back propagates is based on all the parameters of this formula) :
   
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)     
                                                                                                            
   Updating the Q-table based on the newest Q-value :
            
                q_table[discrete_state + (action, )] = new_q
                

            elif new_state[0] >= env.goal_position:
                
                print(f"We made it on episode {episode}")
                
                q_table[discrete_state + (action, )] = 0
                

            discrete_state = new_discrete_state
    
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        env.close()
