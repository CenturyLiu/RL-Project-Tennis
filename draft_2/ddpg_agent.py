#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:14:09 2020

@author: shijiliu
"""


from model import Actor, ActorParamNoise, CentralizedCritic
from param_update import hard_update, add_param_noise
from torch.optim import Adam
import torch
import numpy as np

from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    def __init__(self, num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, lr_actor=1.0e-4, lr_critic=1.0e-4, random_seed = 4, device = device, weight_decay = 0.0):
        super(DDPGAgent, self).__init__()
        
        self.device = device
        self.weight_decay = weight_decay
        
        # create actor/target_actor and critic/target_critic
        self.actor = Actor(local_obs_dim,local_action_size,random_seed).to(self.device)
        self.critic = CentralizedCritic(global_obs_dim , global_action_size).to(self.device)
        self.target_actor = Actor(local_obs_dim,local_action_size,random_seed).to(self.device)
        self.target_critic = CentralizedCritic(global_obs_dim, global_action_size).to(self.device)
        
        #noise
        self.action_noise = OUNoise(local_action_size, scale = 1.0, sigma = 0.1)
        self.param_noise = ActorParamNoise(local_obs_dim, local_action_size,random_seed,stddev = 0.5).to(self.device)
        #self.param_noise_rate = 0.999 # apply this rate to the param noise, gradually get rid of the noise
        #self.use_action_noise = use_action_noise
        #self.use_param_noise = use_param_noise
        
        # copy parameters to target networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
        # create optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=self.weight_decay)
        
    def act(self, obs, action_noise_coef, param_noise_coef):
        obs = torch.FloatTensor(obs).to(self.device)
        #obs = torch.from_numpy(obs).to(self.device)
        #print(obs.dtype)
        self.param_noise.reset_noise_parameters()
        add_param_noise(self.actor, self.param_noise, param_noise_coef)
        action = self.actor(obs) + action_noise_coef * self.action_noise.noise().to(self.device)
        return action
    
    def target_act(self, obs, action_noise_coef, param_noise_coef):
        #obs = torch.FloatTensor(obs).to(self.device)
        #obs = torch.from_numpy(obs).to(self.device)
        self.param_noise.reset_noise_parameters()
        add_param_noise(self.target_actor, self.param_noise, param_noise_coef)
        action = self.target_actor(obs) + action_noise_coef * self.action_noise.noise().to(self.device)
        return action
        