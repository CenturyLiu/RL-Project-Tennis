#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:03:16 2020

@author: shijiliu
"""


from collections import  deque


from unityagents import UnityEnvironment
from maddpg_agent import MADDPG

import numpy as np
import torch
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_env():
    file_location = "/home/shijiliu/self-learning/reinforcement-learning/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/"
    file_name = file_location + "Tennis.x86_64"
    env = UnityEnvironment(file_name=file_name)
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, brain_name, num_agents, action_size, state_size

def training(max_episodes = 3000, episode_length = 1000, random_seed = 4):
    
    
    env, brain_name, num_agents, action_size, state_size = create_env()
    maddpg = MADDPG(num_agents,state_size,action_size,num_agents * state_size, num_agents * action_size, discount_factor = 0.99, tau = 0.001, random_seed = random_seed)
    agent_reward = [[] for _ in range(num_agents)]
    agent_reward_deque = [deque(maxlen=100) for _ in range(num_agents)]
    score_full = []
    score_deque = deque(maxlen = 100)

    for episode in range(1, max_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        obs_full = obs
        
        episode_scores = np.zeros(num_agents)
        
        for episode_t in range(episode_length):
            actions = maddpg.act(obs)
            env_info = env.step(actions)[brain_name]
            
            next_obs = env_info.vector_observations
            next_obs_full = next_obs
            rewards = env_info.rewards
            dones = env_info.local_done
            episode_scores += rewards 
            
            maddpg.step(obs, obs_full, actions, rewards, next_obs, next_obs_full, dones, episode_t)
            
            obs = next_obs
            obs_full = next_obs_full
            
            if np.any(dones):
                break
            
        for i in range(num_agents):
            agent_reward[i].append(episode_scores[i])
            agent_reward_deque[i].append(episode_scores[i])
            
        score_full.append(max(episode_scores))
        score_deque.append(max(episode_scores))
        
        if episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_deque)))
            
        if np.mean(score_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(score_deque)))
            #for i in range(num_agents):
            #    torch.save(agents[i].actor_local.state_dict(), 'checkpoint_actor'+str(i) +'.pth')
            #    torch.save(agents[i].critic_local.state_dict(), 'checkpoint_critic'+str(i)+'.pth')
            torch.save(maddpg.critic.state_dict(),'checkpoint_centralized_critic.pth')
            for i in range(num_agents):
                torch.save(maddpg.actors[i].actor_local.state_dict(), 'checkpoint_actor'+ str(i) + '.pth')
            break
    env.close()
    return maddpg, agent_reward, score_full, random_seed

agents, agent_reward, score_full, random_seed = training(max_episodes = 10000)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score_full)+1), score_full)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()