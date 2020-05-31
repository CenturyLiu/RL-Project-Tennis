#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:04:33 2020

@author: shijiliu
"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from parameter_noise import AdaptiveParamNoise

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, local_obs_dim, action_size, seed = 2, fc1_units = 128, fc2_units = 256):
        """Initialize parameters and build model.
        Params
        ======
            local_obs_dim (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(local_obs_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.batchnorm_1 = nn.BatchNorm1d(fc1_units)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #x = F.relu(self.batchnorm_1(self.fc1(state)))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class ActorParamNoise(Actor):
    def __init__(self, local_obs_dim, action_size, seed = 2, fc1_units = 128, fc2_units = 256):
        Actor.__init__(self,local_obs_dim, action_size, seed = 2, fc1_units = 128, fc2_units = 256)
        
        self.adaptive_noise = AdaptiveParamNoise(initial_stddev = 1.0, desired_action_stddev = 0.2, adoption_coefficient = 1.001)
        
    def reset_with_raw(self, action_non_pertubed, action_pertubed):
        #print("reset with raw")
        #print(action_non_pertubed)
        self.adaptive_noise.update_noise_param(action_non_pertubed, action_pertubed)
        self.reset_noise_parameters()

    def reset_noise_parameters(self):
        self.fc1.weight.data.normal_(std = self.adaptive_noise.current_stddev)
        self.fc2.weight.data.normal_(std = self.adaptive_noise.current_stddev)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)#normal_(std = self.adaptive_noise.current_stddev)
        
class CentralizedCritic(nn.Module):
    """Critic (Value) Model."""
    
    
    def __init__(self, full_obs_dim, action_size, seed = 2, fcs1_units=128, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            full_obs_dim (int): Dimension of the full obsercation, full_obs_dim = n_agents * local_obs_dim
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CentralizedCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        '''
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        '''
        self.fcs1 = nn.Linear(full_obs_dim, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.batchnorm_1 = nn.BatchNorm1d(fcs1_units)
        
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        '''
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        '''
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        '''
        xs = F.leaky_relu(self.fcs1(state))
        #print(xs)
        #print(action)
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        '''
        #xs = F.relu(self.batchnorm_1(self.fcs1(state)))
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x