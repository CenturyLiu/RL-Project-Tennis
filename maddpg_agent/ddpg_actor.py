#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:27:15 2020

@author: shijiliu
"""


from model import Actor, ActorParamNoise
from param_update import hard_update, add_param_noise, soft_update
from torch.optim import Adam
import torch
import numpy as np

from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGActor():
    def __init__(self, num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, lr_actor=1.0e-4,  random_seed = 4, device = device):
        super(DDPGActor, self).__init__()
        
        self.device = device
        
        
        # create actor/target_actor and critic/target_critic
        self.actor_local = Actor(local_obs_dim,local_action_size,random_seed).to(self.device)
        self.actor_target = Actor(local_obs_dim,local_action_size,random_seed).to(self.device)
    
        #noise
        self.action_noise = OUNoise(local_action_size, seed = random_seed, theta = 0.15,sigma = 0.2)
        #self.param_noise = ActorParamNoise(local_obs_dim, local_action_size,random_seed,stddev = 0.5).to(self.device)
        
        # copy parameters to target networks
        hard_update(self.actor_target, self.actor_local)
        
        
        # create optimizers
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        
    def act(self, local_obs, noise_coef, add_noise = True):
        state = torch.from_numpy(local_obs).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.action_noise.sample() * noise_coef#self.sigma * np.random.randn(self.action_size)#self.noise.sample()[0] * self.noise_coef
        return np.clip(action, -1, 1)
    
    def target_act(self, local_obs, noise_coef = 0, add_noise = False):
        #state = torch.from_numpy(local_obs).float().to(device)
        state = local_obs
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_target.train()
        if add_noise:
            action += self.action_noise.sample() * noise_coef#self.sigma * np.random.randn(self.action_size)#self.noise.sample()[0] * self.noise_coef
        return np.clip(action, -1, 1)
        
    def update_target(self, tau):
        soft_update(self.actor_target, self.actor_local, tau)