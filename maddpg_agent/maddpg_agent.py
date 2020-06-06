#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:26:48 2020

@author: shijiliu
"""


import torch
import torch.nn.functional as F
from torch.optim import Adam

import random
import numpy as np

from ddpg_actor import DDPGActor
from param_update import soft_update, hard_update
from model import Critic
from replaybuffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
N_LEARN_UPDATES = 10
TRAIN_EVERY = 2

class MADDPG:
    def __init__(self, num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, discount_factor = 0.95, tau = 0.02, device = device, random_seed = 4, lr_critic = 1.0e-4, weight_decay = 0.0):
        super(MADDPG, self).__init__()
        
        # parameter configuration
        self.num_agents = num_agents
        self.device = device
        self.discount_factor = discount_factor
        self.tau = tau
        self.num_agents = num_agents
        self.global_action_size = global_action_size
        self.global_obs_dim = global_obs_dim
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        
        # define actors
        self.actors = [DDPGActor(num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, device = device) for _ in range(num_agents)]
        # define centralized critic
        self.critic = Critic(global_obs_dim , global_action_size, self.random_seed).to(self.device)
        self.target_critic = Critic(global_obs_dim, global_action_size, self.random_seed).to(self.device)
        hard_update(self.target_critic, self.critic)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=self.weight_decay)
        
        # noise coef
        self.noise_coef = 1.0
        self.noise_coef_decay = 1e-6
        
        # Replay memory
        self.memory = ReplayBuffer( BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, obs_all_agents):
        actions = [ddpg_actor.act(local_obs, self.noise_coef) for ddpg_actor, local_obs in zip(self.actors,obs_all_agents)]
        return actions
        
    def target_act(self, obs_all_agents):
        actions = [ddpg_actor.target_act(local_obs, noise_coef = 0, add_noise = False) for ddpg_actor, local_obs in zip(self.actors,obs_all_agents)]
        return actions
    
    def step(self, obs, obs_full, actions, rewards, next_obs, next_obs_full, dones, timestep):
        self.memory.add(obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)
        
        timestep = timestep % TRAIN_EVERY

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep == 0:
            for _ in range(N_LEARN_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, self.discount_factor)
    
    def learn(self, experiences, gamma):
        obs, obs_full, action, reward, next_obs, next_obs_full, done = experiences
        
        
        obs = obs.permute(1,0,-1) # agent_id * batch_size * state_size
        obs_full = obs_full.view(-1,self.global_obs_dim)
        next_obs = next_obs.permute(1,0,-1)
        next_obs_full = next_obs_full.view(-1,self.global_obs_dim)
        action = action.reshape(-1,self.global_action_size)
        
        # ---------------- update centralized critic ----------------------- #
        self.critic_optimizer.zero_grad()
        
        # get target actions from all target_actors
        target_actions = np.array(self.target_act(next_obs))
        target_actions = torch.from_numpy(target_actions).float().permute(1,0,-1)
        target_actions = target_actions.reshape(-1,self.global_action_size)
        
        # update critic
        with torch.no_grad():
            q_next = self.target_critic.forward(next_obs_full,target_actions.to(self.device))
        
        y = reward + gamma * q_next * (1 - done)
        
        
        q = self.critic.forward(obs_full, action)
        
        critic_loss = 0
        for i in range(self.num_agents):
            critic_loss += F.mse_loss(q, y[:,i].detach().reshape(-1,1)) / self.num_agents
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------- update actor for all agents --------------------- #
        for ii in range(len(self.actors)):
            self.actors[ii].actor_optimizer.zero_grad()
            
            q_action = [ self.actors[i].actor_local(ob) if i == ii \
                   else self.actors[i].actor_local(ob).detach()
                   for i, ob in enumerate(obs) ]
            
            
            q_action = torch.stack(q_action).permute(1,0,-1)
            q_action = q_action.reshape(-1,self.global_action_size).to(self.device)
            
            
            # policy_gradient
            actor_loss = -self.critic.forward(obs_full,q_action).mean()
            actor_loss.backward()
            self.actors[ii].actor_optimizer.step()
            
        # --------------- soft update all target networks ------------------- #
        soft_update(self.target_critic, self.critic, self.tau)
        for actor in self.actors:
            actor.update_target(self.tau)
        
        # -------------- reset noise --------------------------------------- #
        for actor in self.actors:
            actor.action_noise.reset()
        