# Reinforcement Learning

<br>

⚠️ Tutorial refinement in progress. Some codes and guides might look weird. Give us some days to make it right.

<br>

⚠️ Warning : This guide deals deep Q-Learning after a certain point and so the python codes are written assuming that anyone following this is familiar with basic python programming, how to develop models, and matrices manipulation, etc. If you are not upto date on the concepts, please refresh them before jumping in the code. The README, on the other hand, is written very thorougly, explaining all concepts from beginning, and can be easily followed by anyone.

### Contact us :
For any query, please reach us at cserveairf@gmail.com or create a pull request regarding your issue.
<br>
<br>

## Topics
This is a guide to help users get started with reinforcement learning. It will cover the following topics :
<br>
<br>
  ♦️ [Introduction and types of machine learning](#-introduction-and-types-of-machine-learning)<br>
  
  ♦️ [Understanding reinforcement learning](#-understanding-reinforcement-learning)<br>
  
  ♦️ [How does the learning process works](#-how-does-the-learning-process-works)<br>
  
  ♦️ [Concepts of reinforcement learning](#-concepts-of-reinforcement-learning)<br>
  
  ♦️ [Understanding the Markov decision process](#-understanding-markov-decision-process)<br>
  
  ♦️ [Implementation of Q-learning algorithm](#-implementation-of-the-q-learning-algorithm)<br>
<br>

## 🟧 Introduction and types of Machine Learning
The ability of a machine to learn and improve from experience rather than explicit programming is termed as machine learning.

Types of machine learning :<br>

   ✳️ Supervised<br>
  
   ✳️ Unsupervised<br>
  
   ✳️ Reinforcement<br>

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>

## 🟧 Understanding Reinforcement Learning
Reinforcement means encouraging a pattern or behaviour. This form of ML is a _hit and trial_ method, because the model is new to the surroundings. The only way to learn is to experience and then, learn from that experience. 

The main idea is to learn through a "Reward and Punishment" system. For every correct move, there is a reward, for every wrong one, there is a punishment. The model tries to maximise the reward.

It is to be noted that 'training' datasets are not present here. Neither the fed data is classified or labled. The model needs to figure out the path and the best approach to perform a certain task. The environment is also unknown to the agent.

:atom: **Consider an analogy** Let's suppose a baby is learning to walk towards a box of chocolates. The box acts as reward. 2 things can happen here.<br>
First, the baby starts walking and makes it to the box. This result is positive since the goal is achived, and hence the baby is rewarded.<br>
Second, it starts walking but falls due to some obstacle. This is a negative result. Not only the goal isn't achieved, but the baby also got hurt. Hence, we can say that baby is punished.

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>

## 🟧 How does the learning process works?
There are 2 main components of the process, the learing agent, and the environment. The environment, just like in the real world, determines all parameters of larning. In programming, we call that _algorithm_.

The process starts when environment sends a **state** to the agent. The agent then takes an action on it. Based on the action taken by the agent, the environment sends the next **state** and the respective reward to the agent. The agent then updates its knowledge based on the reward and evalutes its previous action on it.<br>
This goes on and on till the environment sends a terminal state. 

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>

## 🟧 Concepts of reinforcement learning
  ⚛️ Reward Maximisation : The agent must be trained in a way, that it takes the _best_ action in order to maximise the reward
  
  ⚛️ Exploration and Exploitation : The agent explores and captures more information about the environment, and then uses that information to highten the rewards

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>

## 🟧 Understanding Markov decision process
This process is a mathematical approach for mapping a solution in reinforcement learning. It can be assumed that, the purpose of reinforcement learning is to solve a Markov decision process. There are a certain number of parameters to get to the solution. They include :<br>
  ☑️ Set of actions (A)<br>
  ☑️ Set of states (S)<br>
  ☑️ Reward (R)<br>
  ☑️ Policy used to approach the problem (pi)<br>
  ☑️ Value (V)<br>
  
The series of actions taken by the agent throughout the process, defines the _policy_. And the collection of rewards recieved defines the _value_. The goal is to maximise the rewards by choosing the optimum policy.

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>

## 🟧 Implementation of the Q-learning algorithm
The idea of Q-learning, is to have "Q-values" for every action that a model takes, given a state. These values need to be updated overtime in such a way, that after a course of actions, _good_ result is produced. It is done by "Rewarding" the agent.<br>
Q-Learning is also known as "Model free learning" because the learning algorithm we write is applicable to any environment.
<br>
<br>
The Q-learning formula : <br>
<img align="left" alt="q-learning" width="1000px" src="https://miro.medium.com/max/2844/1*VItpGaVoIUnh0RUEArqSGQ.png">
<br>

The codes of Q-learning are in continuation. Please proceed serial wise to understand everything, else, it will be very confusing as to why we are doing a certain step. For running the environment, we need a library called gym.<br>
<br>
Installing gym - <br>

    sudo pip install gym

Or -

    sudo python3 -m pip install gym
  
<br>
⚠️ Note : gym needs another library called 'pyglet' to work. Pyglet comes pre-installed with openGL, but it is the outdated version on purpose, since python v2.x only supports pyglet version (<= 1.5.0, >= 1.4.0). If you are using python3(recommended), please update pyglet - <br>
<br>

    sudo pip install pyglet --upgrade

If, however, your machine does not have pyglet installed, please install the latest version for python3 - <br>

    sudo pip install pyglet

Or - 

    sudo python3 -m pip install pyglet

<br>
An example models directory and a logs directory has been attached for reference.<br>
<br>

This tutorial is complete. Be sure to check out other ones too. Enjoy!!!

<br/>
<div align="right">
    <b><a href="#Topics">⬆️ Back to Top</a></b>
</div>
<br/>
