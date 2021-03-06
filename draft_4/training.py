#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:14:39 2020

@author: shijiliu
"""
from unityagents import UnityEnvironment

from ddpg_agent import Agent

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from collections import deque

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

def training():
    # config parameters
    model_dir = '/home/shijiliu/self-learning/reinforcement-learning/deep-reinforcement-learning/p3_collab-compet/draft_3/'
    
    number_of_episodes = 10000
    episode_length = 800
    
    random_play_episodes = 1000
    
    random_seed = np.random.randint(10000)
    
    # create env, get essential env info
    env, brain_name, num_agents, action_size, state_size = create_env()
    
    agent_reward = [[] for _ in range(num_agents)]
    agent_reward_deque = [deque(maxlen=100) for _ in range(num_agents)]
    score_full = []
    score_deque = deque(maxlen = 100)
    
    # create ddpg agents
    agents = [Agent(state_size, action_size, 1, random_seed) for _ in range(num_agents)] # create num_agents ddpg agents
    
    
    for i_episode in range(1,number_of_episodes + 1):
        # reset the environment and get initial observation
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        # reshape states, assume each agent can see global condition
        states = np.reshape(states,(1,-1))
        
    
        # reset ddpg agents
        for agent in agents:
            agent.reset()
        episode_scores = np.zeros(num_agents)
        
        for t in range(episode_length):
            actions = []
            for ii in range(num_agents):
                actions.append(agents[ii].act(states))
            env_actions = np.reshape(np.array(actions),(1,-1))
            
            if i_episode < random_play_episodes:
                env_actions = 2 * np.random.rand(num_agents, action_size) - 1.0
                env_actions = np.clip(env_actions,-1,1)
                env_actions = np.reshape(env_actions,(1,-1))
            
            # play one step
            env_info = env.step(env_actions)[brain_name]
            next_states = env_info.vector_observations
            next_states = np.reshape(next_states,(1,-1))
            rewards = env_info.rewards
            dones = env_info.local_done
            episode_scores += rewards
            
            # store transition, learn if necessary
            for i in range(num_agents):
                agent.step(states, env_actions, rewards[i], next_states, dones[i], t, i)
            
            states = next_states
            
            if np.any(dones):
                break
            
        for i in range(num_agents):
            agent_reward[i].append(episode_scores[i])
            agent_reward_deque[i].append(episode_scores[i])
            
        if i_episode >= random_play_episodes:
            if np.mean(agent_reward_deque[0]) > 1.5 * np.mean(agent_reward_deque[1]):
                agents[1].reset()
                agents[1].noise_coef = 6 # reset noise coefficient to a large number
                agents[0].add_noise = False
                agents[0].update_param = False
                agents[1].add_noise = True
                agents[1].update_param = True
                print("score 0: ",np.mean(agent_reward_deque[0])," score 1: ",np.mean(agent_reward_deque[1]) ," reset 1")
            elif np.mean(agent_reward_deque[1]) > 1.5 * np.mean(agent_reward_deque[0]):
                agents[0].reset()
                agents[0].noise_coef = 6 # reset noise coefficient to a large number
                agents[1].add_noise = False
                agents[1].update_param = False
                agents[0].add_noise = True
                agents[0].update_param = True
                print("score 0: ",np.mean(agent_reward_deque[0])," score 1: ",np.mean(agent_reward_deque[1]) ," reset 0")
            else:
                agents[0].add_noise = True
                agents[0].update_param = True
                agents[1].add_noise = True
                agents[1].update_param = True
            
        score_full.append(max(episode_scores))
        score_deque.append(max(episode_scores))
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_deque)))
            
        if np.mean(score_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_deque)))
            for i in range(num_agents):
                torch.save(agents[i].actor_local.state_dict(), 'checkpoint_actor'+str(i) +'.pth')
                torch.save(agents[i].critic_local.state_dict(), 'checkpoint_critic'+str(i)+'.pth')
            break
    env.close()
    return agents, agent_reward, score_full, random_seed

agents, agent_reward, score_full, random_seed = training()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score_full)+1), score_full)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()