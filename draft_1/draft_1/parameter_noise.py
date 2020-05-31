#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:33:01 2020

@author: shijiliu
"""

# this AdaptiveParamNoise class is adapted from openai/baselines
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
import numpy as np

class AdaptiveParamNoise():
    def __init__(self, initial_stddev = 0.1, desired_action_stddev = 0.1, adoption_coefficient = 1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        
        self.current_stddev = initial_stddev
        
    def adapt(self, distance):
        distance = distance.detach().cpu().numpy()
        #print("distance == ",distance)
        if distance > self.desired_action_stddev:
            # Decrease stddev
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev
            self.current_stddev *= self.adoption_coefficient
        print("distance == ",distance, ", std == ",self.current_stddev)
        return self.current_stddev
        
    def ddpg_distance(self, action_non_pertubed, action_pertubed):
        #print("action_non_pertubed = ",action_non_pertubed)
        #print("action_pertubed = ", action_pertubed)
        #distance = (((action_non_pertubed - action_pertubed)**2).mean())**0.5
        distance = (((action_non_pertubed - action_pertubed)**2).mean() * len(action_non_pertubed)  / (len(action_non_pertubed) - 1))**0.5 # estimate of stddev
        return distance
        
    def update_noise_param(self,action_non_pertubed,action_pertubed):
        distance = self.ddpg_distance(action_non_pertubed,action_pertubed)
        stddev = self.adapt(distance)
        return stddev