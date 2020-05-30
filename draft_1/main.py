#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:04:14 2020

@author: shijiliu
"""


from unityagents import UnityEnvironment
import numpy as np
import torch
from maddpg_ensemble import MADDPG_ENSEMBLE
from collections import namedtuple, deque
import matplotlib.pyplot as plt

file_location = "/home/shijiliu/self-learning/reinforcement-learning/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/"
file_name = file_location + "Tennis.x86_64"
env = UnityEnvironment(file_name=file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

BUFFER_SIZE = int(1e6)
NUM_SUB_POLICY = int(5)
BATCH_SIZE = 64
N_LEARN_UPDATES = 10     # sample N_LEARN_UPDATES batches during training

max_episode = 3000
max_steps = 1000

def run(max_episode, max_steps, batch_size):
    scores_deque = deque(maxlen=100)
    episode_rewards = []
    for episode in range(max_episode):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations     
        #reset OUNoise
        for i in range(maddpg_train.num_agents):
            maddpg_train.agents[i].reset()
                
        scores = np.zeros(num_agents)
        for step in range(max_steps):
            actions, sub_policy_id = maddpg_train.get_actions(states)
            action_np = [a.detach().cpu().numpy() for a in actions]
            env_info = env.step(action_np)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards 
            
        
            if all(dones) or step == max_steps - 1:
                dones = [1 for _ in range(maddpg_train.num_agents)]
                maddpg_train.replay_buffer[sub_policy_id].push(states, [a.detach() for a in actions], rewards, next_states, dones)
                episode_rewards.append(max(scores))
                scores_deque.append(max(scores))
                print("episode: {}  |  reward: {}  \n".format(episode, np.round(scores, decimals=4)))
                break
            else:
                dones = [0 for _ in range(maddpg_train.num_agents)]
                maddpg_train.replay_buffer[sub_policy_id].push(states, [a.detach() for a in actions], rewards, next_states, dones)
                states = next_states 
                if len(maddpg_train.replay_buffer[sub_policy_id]) > 20 * batch_size:
                    for _ in range(N_LEARN_UPDATES):
                        maddpg_train.update(sub_policy_id)
             
        #print(np.mean(scores_deque))        
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            break
            
        
    return episode_rewards



maddpg_train = MADDPG_ENSEMBLE(num_agents, state_size, action_size, buffer_maxlen = BUFFER_SIZE, num_sub_policy = NUM_SUB_POLICY, batch_size = BATCH_SIZE)
scores = run(max_episode,max_steps,BATCH_SIZE)


env.close()
#plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()