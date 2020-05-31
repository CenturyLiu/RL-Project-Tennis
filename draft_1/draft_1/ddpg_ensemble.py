#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:09:46 2020

@author: shijiliu
"""


import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, CentralizedCritic, ActorParamNoise

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import MultiAgentReplayBuffer


GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
#TRAIN_EVERY = 20        # every TRAIN_EVERY timesteps, update the network 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_ENSEMBLE():
    def __init__(self, number_agents, obs_dim, action_dim, agent_id, actor_lr=LR_ACTOR, critic_lr=LR_CRITIC, gamma=GAMMA, tau=TAU ,num_sub_policy = 3):
        '''
        

        Parameters
        ----------
        number_agents : int
            the number of agents.
        obs_dim : int
            the observation dimension of each agent
        action_dim : int
            action dimension of each agent
        agent_id : int
            id of an agent
        actor_lr : float, optional
            learning rate of the actor. The default is LR_ACTOR.
        critic_lr : float, optional
            learning rate of the critic. The default is LR_CRITIC.
        gamma : float, optional
            discount factor. The default is GAMMA.
        tau : float, optional
            parameter for softupdate of the target network. The default is TAU.
        num_sub_policy : int, optional
            number of sub-policies for ensemble. The default is 3.

        Returns
        -------
        None.

        '''
        self.agent_id = agent_id
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = number_agents
        self.num_sub_policy = int(num_sub_policy)
        
        self.device = device
        
        # define critic
        self.critic_input_dim = int(np.sum([obs_dim for agent in range(number_agents)]))
        self.critic = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim, self.action_dim * self.num_agents).to(self.device)
        
        # define num_sub_policy agents for ensemble
        self.actor_input_dim = self.obs_dim
        self.actor = [Actor(self.actor_input_dim, self.action_dim).to(self.device) for _ in range(self.num_sub_policy)]
        self.actor_target = [Actor(self.actor_input_dim, self.action_dim).to(self.device) for _ in range(self.num_sub_policy)]
        
        # define num_sub_policy adaptive noise agent, one to one corresponding to the sub-policies
        self.param_noise = [ActorParamNoise(self.actor_input_dim, self.action_dim).to(self.device) for _ in range(self.num_sub_policy)]
        
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        for i in range(self.num_sub_policy):
            for target_param, param in zip(self.actor_target[i].parameters(), self.actor[i].parameters()):
                target_param.data.copy_(param.data)
                
        # define loss function and optimizer
        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = [optim.Adam(self.actor[i].parameters(), lr=actor_lr) for i in range(self.num_sub_policy)]
        
        # noise
        self.noise = [OUNoise(self.action_dim,theta=0.1,sigma=1.0) for i in range(self.num_sub_policy)]
        self.add_action_noise = False
        
        # memory
        #self.memory = [MultiAgentReplayBuffer(self.num_agents,BUFFER_SIZE) for _ in range(self.num_sub_policy)]
        
    def get_action(self, state, sub_policy_id):
        '''
        

        Parameters
        ----------
        state : float array
            observation state for the agent.

        Returns
        -------
        action : tensor (float)
            the tensor of the float.
        sub_policy_id : int
            the id of the subpolicy we choose.

        '''
        #sub_policy_id = random.choice(np.arange(self.num_sub_policy))
        state = torch.autograd.Variable(torch.from_numpy(state).float()).to(self.device)
        #print(state)
        self.param_noise[sub_policy_id].reset_noise_parameters()
        self.add_param_noise(sub_policy_id) # add parameter noise to the agent's parameters 
        
        action = self.actor[sub_policy_id].forward(state)
        if self.add_action_noise:
            action += torch.tensor(self.noise[sub_policy_id].noise()).float().to(self.device)#torch.tensor(np.random.randn(self.action_dim)).float().to(self.device)
        
        action = torch.clamp(action,-1,1)
        return action
    

        
    def learn(self, sub_policy_id,indiv_reward_batch, indiv_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions):
        indiv_reward_batch = torch.FloatTensor(indiv_reward_batch).to(self.device)
        indiv_reward_batch = indiv_reward_batch.view(indiv_reward_batch.size(0), 1).to(self.device) 
        indiv_obs_batch = torch.FloatTensor(indiv_obs_batch).to(self.device)          
        global_state_batch = torch.FloatTensor(global_state_batch).to(self.device)    
        global_actions_batch = torch.stack(global_actions_batch).to(self.device)      
        global_next_state_batch = torch.FloatTensor(global_next_state_batch).to(self.device)
        

        # update critic        
        self.critic_optimizer.zero_grad()
        
        curr_Q = self.critic.forward(global_state_batch, global_actions_batch)
        next_Q = self.critic_target.forward(global_next_state_batch, next_global_actions)
        estimated_Q = indiv_reward_batch + self.gamma * next_Q
        
        critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer[sub_policy_id].zero_grad()

        policy_loss = -(self.critic.forward(global_state_batch, global_actions_batch).mean()) * (1 / self.num_sub_policy)
        curr_pol_out = self.actor[sub_policy_id].forward(indiv_obs_batch)
        policy_loss += -(curr_pol_out**2).mean() * 1e-3 
        policy_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer[sub_policy_id].step()
        
        self.target_update(sub_policy_id)
        
        # update param noise
        # first add noise
        
        self.param_noise[sub_policy_id].reset_noise_parameters() # generate new params with current std
        self.add_param_noise(sub_policy_id)
        action_with_noise = self.actor[sub_policy_id].forward(indiv_obs_batch)
        
        # then extract noise
        self.extract_param_noise(sub_policy_id)
        action_without_noise = self.actor[sub_policy_id].forward(indiv_obs_batch)
        
        #print("updating")
        self.param_noise[sub_policy_id].reset_with_raw(action_with_noise,action_without_noise)
        #print("sub_policy_id = ",sub_policy_id," stddev = ",self.param_noise[sub_policy_id].adaptive_noise.current_stddev)
        
    def target_update(self,sub_policy_id):
        for target_param, param in zip(self.actor_target[sub_policy_id].parameters(), self.actor[sub_policy_id].parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
    def add_param_noise(self, sub_policy_id):
        for noise_param, local_param in zip(self.actor[sub_policy_id].parameters(), self.param_noise[sub_policy_id].parameters()):
            noise_param.data.copy_(local_param.data +  noise_param.data)
            
    def extract_param_noise(self, sub_policy_id):
        # for updating parameter noise stddev, first add current noise, then extract the noise
        for noise_param, local_param in zip(self.actor[sub_policy_id].parameters(), self.param_noise[sub_policy_id].parameters()):
            noise_param.data.copy_(local_param.data -  noise_param.data)
    
    def reset(self):
        for noise in self.noise:
            noise.reset()
        
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()