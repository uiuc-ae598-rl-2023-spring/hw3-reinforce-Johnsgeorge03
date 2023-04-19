#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:31:12 2023

@author: john
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from agent_network import *

hidden_size = 100
gamma       = 1.0
lr          = 1e-3
device      = 'cpu'
opt_type    = 'sgd'
n_train_eps = 50000


## DIRECTORIES
import os
test_dir            = 'test7/'
weight_dir          = test_dir + 'NN_weights/' + opt_type + '/'
fig_dir             = test_dir + 'figures/' + opt_type +'/'
reward_dir          = test_dir + 'rewards/'
reward_file         = 'rewards_' + opt_type +'.txt'
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(reward_dir, exist_ok=True)

## TRAIN
env         = gridworld.GridWorld(hard_version=False)
agent       = Agent(env, hidden_size, lr, gamma, device, opt_type)
all_rewards = agent.reinforce(n_train_eps, 1, weight_dir)




## WRITE REWARDS AND MODEL PARAMS TO FILES
torch.save(agent.policy_network.state_dict(), 
           weight_dir + 'pnet_model_weights_end.pth')
with open(reward_dir + reward_file, 'w+') as f:
    for items in all_rewards:
        f.write('%s\n' %items)
    print("File written successfully")
f.close()


## PLOTS
import plotter_fns
label_list    = [opt_type]
reward_list   = [reward_dir + reward_file]# reward_dir + 'rewards_sgd.txt']
weight_file   = weight_dir + 'pnet_model_weights_max_mean.pth'
plotter       = plotter_fns.Plotter(env, weight_file, hidden_size, device)

plotter.plot_learning_curve(gamma, reward_list, 
                            fig_dir + 'learning_curve.png',
                            label_list)
if os.path.exists(reward_dir + 'rewards_adam.txt') and \
    os.path.exists(reward_dir + 'rewards_sgd.txt'):
    if label_list[0] == 'sgd' and len(label_list) == 1:
        label_list.append('adam')
        reward_list.append(reward_dir + 'rewards_adam.txt')
    else:
        label_list.append('sgd')
        reward_list.append(reward_dir + 'rewards_sgd.txt')
    
    plotter.plot_learning_curve(gamma, reward_list, 
                                test_dir + 'figures/' + 'learning_curve_all.png',
                                label_list)

log = plotter.plot_trajectory(fig_dir + 'trajectory.png')
plotter.plot_policy(fig_dir + 'policy.png')

test_net = PGN(1, hidden_size, env.num_actions).to(device)
test_net.load_state_dict(torch.load(weight_file))
P_mat = torch.zeros(25, 4)
for i in range(25):
    state  = np.array(i)
    state  = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs  = test_net.forward(state).cpu()
    P_mat[i] = probs
    
plotter.plot_all_policy(P_mat.detach().numpy(), fig_dir + 'all_policy.png')
## VALUE FUNCTIONS
# tol      = 1e-8
# max_iter = 4000
# Vl_vi    = []
# V_star_vi, P_star_vi = value_iteration(env, gamma, tol, max_iter, Vl_vi)

# print("value_iter:\n ", V_star_vi)
# V_sarsa       = TD0(env, test_net, 0.5, 1.0, 2000, device)
# print("reinforce:\n ",V_sarsa)

