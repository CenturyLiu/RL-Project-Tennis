#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:40:46 2020

@author: shijiliu
"""
import torch

from ddpg_agent import DDPGAgent
from param_update import soft_update


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, discount_factor = 0.95, tau = 0.02, device = device):
        super(MADDPG, self).__init__()
        
        # store configuration parameters
        self.device = device
        self.discount_factor = discount_factor
        self.tau = tau
        self.num_agents = num_agents
        
        # create maddgp agent
        self.maddpg_agent = [DDPGAgent(num_agents, local_obs_dim, local_action_size, global_obs_dim, global_action_size, device = self.device) for _ in range(num_agents)]
        
        # iteration counter
        self.iter = 0
        
    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors
        
    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors
    
    def act(self, obs_all_agents, action_noise_coef = 0.0, param_noise_coef = 0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, action_noise_coef, param_noise_coef) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions
    
    def target_act(self, obs_all_agents, action_noise_coef = 0.0, param_noise_coef = 0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, action_noise_coef, param_noise_coef) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples
        
        obs = torch.FloatTensor(obs).to(self.device)
        obs_full = torch.FloatTensor(obs_full).to(self.device)
        obs_full = [ob.view(1,-1) for ob in obs_full]
        #next_obs_full = [torch.FloatTensor(next_ob).to(self.device) for next_ob in next_obs_full]
        
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        next_obs1 = []
        for i in range(self.num_agents):
            next_obs1.append(next_obs[:,i])
        #print('next_obs.size() = ',next_obs.size())
        
        next_obs_full = torch.FloatTensor(next_obs_full).to(self.device)
        next_obs_full = [next_ob.view(1,-1) for next_ob in obs_full]
        
        
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        
        # update critic network with MSE of time difference
        agent.critic_optimizer.zero_grad()
        
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs1)
        #print('target_actions.size() == ',len(target_actions))
        target_actions = torch.cat(target_actions, dim=1)
        #print(target_actions)
        
        target_critic_obs = next_obs_full[:,0]
        target_critic_obs = target_critic_obs.to(self.device)
        target_critic_actions = target_actions.to(device)
        
        #print('target_critic_actions.size == ',target_critic_actions.size())
        #print('target_critic_obs.size == ',target_critic_obs.size())
        #target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic.forward(target_critic_obs,target_critic_actions)
            
        
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        y = reward[:,agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:,agent_number].view(-1, 1))
        
        # get Q(s,a)
        
        critic_action = torch.stack([torch.stack(act).view(1,-1) for act in action])
        critic_action = critic_action[:,0]
        #print(critic_action.size())
        #critic_action = torch.cat(action, dim=1).to(device)
        critic_obs = obs_full[:,0]
        q = agent.critic.forward(critic_obs.detach(), critic_action.detach())
        
        # update critic network
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        # update actor network with policy gradient
        agent.actor_optimizer.zero_grad()
        
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        obs1 = []
        for i in range(self.num_agents):
            obs1.append(obs[:,i])
        
         
        
        q_action = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs1) ] 
        
        
        q_action = torch.cat(q_action, dim=1).to(device)
        
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        #q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        #q_obs = obs_full.t().to(device)
        
        q_obs = obs_full[:,0]
        
        # get the policy gradient
        actor_loss = -agent.critic(q_obs,q_action).mean()
        actor_loss.backward()
        
        agent.actor_optimizer.step()
        
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        return al, cl
            
        
    
    
    
    
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)