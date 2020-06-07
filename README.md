# RL-Project-Tennis
Train a pair of agent to solve the Tennis environment

## Part 1: Environment Introduction
The environment of this project is similar but not identical to the [Unity Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

![Trained agents](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

Two trained agents playing tennis by controlling rackets to bounce a ball over a net.
[Image source](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

- Observation space and action space

  Number of agents:  2
  
  Observation space: A single observation is a 8-variable array corresponding to the position and velocity of the ball and racket. 3 single observations are stacked together to form the stacked-observation at each environment step. Each agent receives its own stacked-observation.
  
  Action space: 2 continuous actions for each agent. One action corresponds to move towards to / away from the net. The other corresponds to jump.  


- Reward setup
  The task is episodic, agents receive rewards during the episode.

  |Condition|Reward|
  |---------|------|
  |agent hits the ball over the net|+0.1|
  |ball hit the ground|-0.01|
  |agent hits the ball out of bounds|-0.01|
  
  Each agent receives its own reward during episodes. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores and get the **score** of the episode.
  
  The Tennis is considered solved if the average episode **score** over 100 consecutive episodes achieves 0.5. 
  


## Part 2: Getting started


## Part 3: Idea for solving the task

## Part 4: Repository code usage

## Part 5: Demo for trained agent
