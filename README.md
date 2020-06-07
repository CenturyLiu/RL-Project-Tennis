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
   0. Install python dependencies based on the instruction on [udacity deep reinforcement learning github repo](https://github.com/udacity/deep-reinforcement-learning)
   
   1. Install Tennis environment
       - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
       
       - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
       
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
      
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    
   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
   

## Part 3: Idea for solving the task
  I approached the Tennis environment based on 2 different methods.
  - Self play method
  
    Create a single ddpg network. The action of both tennis agents are chosen from this network. 
    
  - Multi-agent method
  
    Create a maddpg agent with 2 seperate actor-networks. 
    
   For task-solving detail, see the [Report](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/tennis%20project%20report.pdf). 

## Part 4: Repository code usage
  This repository contains 6 different solution packages, please disregard the "draft" packages, which are failed trials. The useful packages are "self-play", "self-play-test" and "maddpg_agent", each one of these 3 packages contains a successful solution that can directly be used to solve the environment.
  
  | solution package | model file | agent file(s) | main file| saved weights |
  |------------------|------------|---------------|----------|---------------|
  |self_play| model.py | ddpg_agent.py (Replaybuffer and OUNoise included in the same file)|training.py| actor: checkpoint_actor.pth;                                     critic: checkpoint_critic.pth| 
  |self-play-test| model.py | ddpg_agent.py (Replaybuffer and OUNoise included in the same file)| training.py| actor: checkpoint_actor.pth;  critic: checkpoint_critic.pth|
  |maddpg_agent| model.py|ddpg_actor.py; maddpg_agent.py; replaybuffer.py; OUNoise.py; param_update.py|training.py|actor 0: checkpoint_actor0.pth; actor 1: checkpoint_actor1.pth; critic: checkpoint_centralized_critic.pth|
  
  *Note: remember to change the "file_location" to your location of storing the Tennis environment before use. file_location is defined in function "create_env()" in the main file for all 3 packages.* 
  
  To see the difference of self_play and self-play-test, please refer to [Report](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/tennis%20project%20report.pdf).
  
  If your agent just cannot solve the environemnt (hopefully that's not the case), my [Report](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/tennis%20project%20report.pdf) includes my **hypothesis** about adding **batchnorm layer** in the agent model, which may lead to agent's **not able to solve** the environment even though the other parts of the code is correct.

## Part 5: Demo for trained agent
![Demo of my trained maddpg agents](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/plots/tennis_demo.gif)
> Demo of the traing maddpg agents

![self_play](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/plots/solution_238_gamma095.png)
> self_play agent solves the Tennis environment in 238 episodes

![maddpg_agent](https://github.com/CenturyLiu/RL-Project-Tennis/blob/master/plots/solution_maddpg_1225.png)
> maddpg_agents solve the Tennis environment in 1225 episodes

## Part 6: References
|reference|reason|
|--------------|------|
| [maddpg algorithm](https://towardsdatascience.com/openais-multi-agent-deep-deterministic-policy-gradients-maddpg-9d2dad34c82)|better understand the maddpg algorithm|
|[nunesma's reinfocement learning file](https://github.com/nunesma/reinforcement_learning/tree/master/p3_collab-compet)| referenced for my implementation of self-play|
|udacity maddpg lab|referenced for my implementation of maddpg|
