#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:32:45 2020

@author: shijiliu
"""
from collections import  deque


from unityagents import UnityEnvironment
from PriotizedReplayBuffer import PrioritizedReplayMemory
from maddpg_agent import MADDPG

import numpy as np
import torch
import os
import matplotlib.pyplot as plt

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

def train():
    
    # config parameters
    model_dir = '/home/shijiliu/self-learning/reinforcement-learning/deep-reinforcement-learning/p3_collab-compet/draft_2/'
    
    number_of_episodes = 10000
    episode_length = 80
    batchsize = 128
    
    t = 0
    
    action_noise_coef = 10.0
    param_noise_coef = 0.0
    action_noise_reduction = 0.9999
    param_noise_reduction = 0.9999
    
    episode_per_update = 2
    
    # create env, get essential env info
    env, brain_name, num_agents, action_size, state_size = create_env()
    
    buffer = PrioritizedReplayMemory(1000*episode_length,alpha = 0.5, beta_start = 0.4)
    
    # initialize policy and critic
    maddpg = MADDPG(num_agents,state_size,action_size,num_agents * state_size, num_agents * action_size, discount_factor = 0.99, tau = 0.001)
    agent_reward = [[] for _ in range(num_agents)]
    score_full = []
    score_deque = deque(maxlen = 100)
    
    # training loop
    for episode in range(number_of_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        obs_full = obs
        
        episode_scores = np.zeros(num_agents)
        
        for episode_t in range(episode_length):
            actions = maddpg.act(obs, action_noise_coef, param_noise_coef)
            action_noise_coef *= action_noise_reduction
            param_noise_coef *= param_noise_reduction
            
            # process the output action to interact with the environment
            action_np = [a.detach().cpu().numpy() for a in actions]
            
            # step the environment for 1 step
            env_info = env.step(action_np)[brain_name]
            
            next_obs = env_info.vector_observations
            next_obs_full = next_obs
            rewards = env_info.rewards
            dones = env_info.local_done
            episode_scores += rewards 
            
            # add data to buffer
            transition = (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)
            
            buffer.push(transition)
            
            obs = next_obs
            obs_full = next_obs_full
            
            if np.any(dones):
                break
        
        # update the networks once after every episode_per_update
        if buffer.storage_size() > batchsize and episode % episode_per_update == 0:
            for a_i in range(num_agents):
                samples,_,_ = buffer.sample(batchsize)
                #print(len(samples))
                ordered_samples = zip(*samples)
                maddpg.update(ordered_samples, a_i)
            maddpg.update_targets() #soft update the target network towards the actual networks
            
        for i in range(num_agents):
            agent_reward[i].append(episode_scores[i])
            
        score_full.append(max(episode_scores))
        score_deque.append(max(episode_scores))
        
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_deque)))
            
        if np.mean(score_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(score_deque)))
            # save models
            save_dict_list = []
            for i in range(num_agents):
                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)
            torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
            
            break
        
        
    env.close()
    return maddpg, agent_reward, score_full

maddpg, agent_reward, score_full = train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(score_full)+1), score_full)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()