#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:49:41 2020

@author: shijiliu
"""
import torch
import numpy as np
import random

from ddpg_ensemble import DDPG_ENSEMBLE
from utils import MultiAgentReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128       # minibatch size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_ENSEMBLE():
    def __init__(self,number_agents, obs_dim, action_dim, buffer_maxlen = BUFFER_SIZE, num_sub_policy = 3, batch_size = BATCH_SIZE):
        self.num_agents = number_agents
        self.num_sub_policy = int(num_sub_policy)
        self.replay_buffer = [MultiAgentReplayBuffer(self.num_agents, buffer_maxlen) for _ in range(self.num_sub_policy)]
        self.agents = [DDPG_ENSEMBLE(number_agents, obs_dim, action_dim, i, num_sub_policy = self.num_sub_policy) for i in range(self.num_agents)]
        self.subpolicy_array = np.arange(self.num_sub_policy)
        self.batch_size = batch_size
        
    def get_actions(self, states):
        actions = []
        sub_policy_id = random.choice(self.subpolicy_array)
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i],sub_policy_id)
            actions.append(action)
        return actions, sub_policy_id
    
    def update(self, sub_policy_id):
        #print("sub_policy_id == ", sub_policy_id)
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
            global_state_batch, global_actions_batch, global_next_state_batch, done_batch = self.replay_buffer[sub_policy_id].sample(self.batch_size)
            
            
        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]
            #print(next_obs_batch_i)
            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.FloatTensor(next_obs_batch_i)
                #print(next_obs_batch_i)
                indiv_next_action = agent.actor[sub_policy_id].forward(next_obs_batch_i.to(device))
                indiv_next_action = [act for act in indiv_next_action]
                indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)
            next_global_actions = torch.cat([next_actions_i for next_actions_i in next_global_actions], 1)
            
        
            self.agents[i].learn(sub_policy_id,indiv_reward_batch_i, obs_batch_i, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions)