#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:54:30 2020

@author: shijiliu
"""
import random
from collections import deque, namedtuple
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "obs_full", "actions", "rewards", "next_obs", "next_obs_full", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, obs, obs_full, actions, rewards, next_obs, next_obs_full, dones):
        """Add a new experience to memory."""
        e = self.experience(obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        obs = torch.from_numpy(np.array([e.obs for e in experiences if e is not None])).float().to(device)
        obs_full = torch.from_numpy(np.array([e.obs_full for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.rewards for e in experiences if e is not None])).float().to(device)
        next_obs = torch.from_numpy(np.array([e.next_obs for e in experiences if e is not None])).float().to(device)
        next_obs_full = torch.from_numpy(np.array([e.next_obs_full for e in experiences if e is not None])).float().to(device)
        
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (obs, obs_full, actions, rewards, next_obs, next_obs_full, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)