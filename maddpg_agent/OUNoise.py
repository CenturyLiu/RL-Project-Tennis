import numpy as np
import torch
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        '''
        mat = np.zeros((self.size))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                mat[i][j] = random.random()
        '''
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(self.size[0]*self.size[1])]).reshape((self.size[0],self.size[1]))
        
        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size) #the random term should follow Weiner process, each imcrements should 
                                                                                   #follow normal distribution with mean == 0
        self.state = x + dx
        return self.state