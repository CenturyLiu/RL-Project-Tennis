import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic, ActorParamNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
TRAIN_EVERY = 1        # every TRAIN_EVERY timesteps, update the network 
DECAY_NOISE = 0.9995         # the noise will decay per episode
N_LEARN_UPDATES = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_param_noise(target, source, rate):
    """
    Add parameter space noise 
    Inputs:
        copy base parameter to model with random noise
        target (torch.nn.Module): Net with random noise
        source (torch.nn.Module): Net with base parameter
        rate (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(rate * target_param.data + param.data)


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(2 * state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(2 * state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(2 * state_size, 2 * action_size, random_seed).to(device)
        self.critic_target = Critic(2 * state_size, 2 * action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((num_agents,action_size), random_seed, mu=0., theta=0.15, sigma=0.1)
        self.add_noise = True
        
        self.param_noise = ActorParamNoise(2 * state_size, action_size, random_seed).to(device)
        self.add_param_noise = True

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        self.noise_coef = 6
        
        # update parameter or not
        self.update_param = True
        
        
    
    def step(self, state, action, reward, next_state, done, timestep, agent_id):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        timestep = timestep % TRAIN_EVERY

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep == 0 and self.update_param:
            for _ in range(N_LEARN_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_id)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        self.param_noise.eval()
        with torch.no_grad():
            self.param_noise.reset_noise_parameters()
            add_param_noise(self.param_noise, self.actor_local, 1)
            #action = self.actor_local(state).cpu().data.numpy()
            action = self.param_noise(state).cpu().data.numpy()
        self.param_noise.train()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample() * self.noise_coef
            self.noise_coef *= DECAY_NOISE
            if self.noise_coef  < 0.01:
                self.noise_coef = 0.01
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        
        
        if agent_id == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim = 1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim = 1)
        
        
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # clip gradient
        #torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        if agent_id == 0:
            actions_pred = torch.cat((actions_pred,actions[:,2:]), dim = 1)
        else:
            actions_pred = torch.cat((actions[:,:2],actions_pred), dim = 1)
        
        
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def copy_parameter(self, other_ddpg_agent):
        self.soft_update(other_ddpg_agent.actor_local, self.actor_local, 1)
        self.soft_update(other_ddpg_agent.critic_local, self.critic_local, 1)
        self.soft_update(other_ddpg_agent.actor_target, self.actor_target, 1)
        self.soft_update(other_ddpg_agent.critic_target, self.critic_target, 1)

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
        
        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.size) #the random term should follow Weiner process, each imcrements should 
                                                                                   #follow normal distribution with mean == 0
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
