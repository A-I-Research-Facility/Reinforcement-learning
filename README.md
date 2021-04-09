# Reinforcement Learning Introduction

### Contact us :
For any query, please reach us at cserveairf@gmail.com or create a pull request regarding your issue.

***

This is a guide to help users get started with reinforcement learning. It will cover the following topics :
<br>
<br>
  ▶️ Introduction and types of machine learning<br>
  ▶️ Understanding reinforcement learning<br>
  ▶️ How does the learning process works<br>
  ▶️ Concepts of reinforcement learning (Reward maximization, Exploration & Exploitation<br>
  ▶️ Understanding the Markov's decision process<br>
  ▶️ Implementation of Q-learning algorithm
 <br>
***
<br>

## ⚫Introduction and types of Machine Learning
Ability of a machine to learn and improve from experience rather than explicit programming.

Types of machine learning :<br>
  🟢 Supervised<br>
  🟢 Unsupervised<br>
  🟢 Reinforcement<br>
<br>

## ⚫Understanding Reinforcement Learning
Reinforcement means encouraging a pattern or behaviour. This form of ML is a _hit and trial_ method, because the model is new to the surroundings. The only way to learn is to experience and then, learn from that experience. 

The main idea is to learn through a "Reward and Punishment" system. For every correct move, there is a reward, for every wrong one, there is a punishment. The model tries to maximise the reward.

It is to be noted that 'training' datasets are not present here. Neither the fed data is classified or labled. The model needs to figure out the path and the best approach to perform a certain task. The environment is also unknown to the agent.

:atom: **Consider an analogy** Let's suppose a baby is learning to walk towards a box of chocolates. The box acts as reward. 2 things can happen here.<br>
First, the baby starts walking and makes it to the box. This result is positive since the goal is achived, and hence the baby is rewarded.<br>
Second, it starts walking but falls due to some obstacle. This is a negative result. Not only the goal isn't achieved, but the baby also got hurt. Hence, we can say that baby is punished.
<br>
<br>

## ⚫How does the learning process works
There are 2 main components of the process, the learing agent, and the environment. The environment, just like in the real world, determines all parameters of larning. In programming, we call that _algorithm_.

The process starts when environment sends a **state** to the agent. The agent then takes an action on it. Based on the action taken by the agent, the environment sends the next **state**.<br>
This goes on and on till the required result is achieved.
