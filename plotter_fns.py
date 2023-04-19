#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:10:34 2023

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.distributions import Categorical

from agent_network import *

class Plotter:
    def __init__(self, env, weight_file, hidden_size, device):
        self.env = env
        self.NN  = PGN(1, hidden_size, env.num_actions).to(device)
        self.NN.load_state_dict(torch.load(weight_file))
        self.device = device
        
        
    def policy(self, state):
        with torch.no_grad():
            state  = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs  = self.NN(state).cpu()
            m      = Categorical(probs)
            action = m.sample()
        return action.item() #probs.argmax().item()  #
    
        
    def plot_learning_curve(self, gamma, source_files, dest_file, label_list):
        DF = []
        for source_file in source_files:
            with open(source_file) as f:
                lines = f.readlines()
    
           
            lines = [line.strip() for line in lines]
            episode_list = [[int(num) for num in line[1:-1].split(',')] 
                            for line in lines]
            num_eps              = len(episode_list)
            undiscounted_returns = np.zeros(num_eps)
            for e in range(num_eps):
                u_returns = 0
                for i in range(len(episode_list[e])):
                    u_returns += episode_list[e][i]
                
                undiscounted_returns[e] = u_returns
            
            mean_ud_returns = []
            sd_ud_returns   = []
            
           
            for i in range(num_eps):
                if i < 100:
                    mean_ud_returns.append(np.mean(undiscounted_returns[0:i+1]))
                    sd_ud_returns.append(np.std(undiscounted_returns[0:i+1]))
                    
        
                    
                else:
                    mean_ud_returns.append(np.mean(undiscounted_returns[i-100:i]))
                    sd_ud_returns.append(np.std(undiscounted_returns[i-100:i]))
                    
     
            row_means = np.array(mean_ud_returns)
            row_stds  = np.array(sd_ud_returns)
    
            # create a pandas DataFrame with columns for x-axis, y-axis, lower bound, and upper bound
            df = pd.DataFrame({'x': range(num_eps),
                                'y': row_means,
                                'lower': row_means - row_stds,
                                'upper': row_means + row_stds})
            
            DF.append(df)
            
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 6))
        # plot the mean values with a variance band using Seaborn's lineplot
        color_list = ['r', 'b']
        for i in range(len(DF)):
            sns.lineplot(data=DF[i], x='x', y='y', ci='sd', linewidth = 2,
                         color= color_list[i],label = label_list[i])
            plt.fill_between(DF[i]['x'], DF[i]['lower'], DF[i]['upper'], alpha=0.25, 
                             color = color_list[i])
    
        # plot the variance band as a shaded area
        
        plt.grid("True")
        plt.legend(loc="best")
        plt.xlabel('Episodes', fontsize = 15)
        plt.ylabel('Mean Undiscounted Returns', fontsize = 15)
        plt.title("Reinforce Learning Curve", fontsize = 20)
        plt.tight_layout()
        plt.savefig(dest_file)
        plt.show()
        plt.close()
                
            
    def plot_trajectory(self, filename):
        s = self.env.reset()

        # Create log to store data from simulation
        log = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }

        # Simulate until episode is done
        done = False
        while not done:
            s = np.array(s)
            a = self.policy(s)
            (s, r, done) = self.env.step(a)
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)

        # Plot data and save to png file
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        fig, ax1 = plt.subplots(3,1, figsize=(12,12), sharex=True,
                                gridspec_kw={'height_ratios': [1, 1, 1]})
        ax1[0].plot(log['t'], log['s'])
        ax1[0].set_ylabel('State', fontsize = 15)
        #ax1[0].set_xlabel('Time')
        ax1[1].plot(log['t'][:-1], log['a'])
        ax1[1].set_ylabel('Action', fontsize = 15)
        #ax1[1].set_xlabel('Time')
        ax1[2].plot(log['t'][:-1], log['r'])
        ax1[2].set_ylabel('Reward', fontsize = 15)
        # ax1[2].set_xlabel('Time')
        plt.suptitle("Trajectory", fontsize = 20)
        plt.xlabel('Time', fontsize = 15)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        plt.close()
        return log
        
    def plot_policy(self, filename):
        Vl_vi    = []
        V_star_vi, P_star_vi= value_iteration(self.env, 1.0, 1e-8, 4000, Vl_vi)
        self.p_star = P_star_vi
        self.v_star = V_star_vi
        fig, ax = plt.subplots(1,1, figsize=(8, 8))
        row     = 10
        counter = 0
        for i in range(5):  # row
            col = 0
            for j in range(5):  # col
                a = self.policy(np.array(counter))
                a_s = self.p_star[counter]
                # get coords
    
                if a == 0:  # right
                    x = col+0.5
                    y = row-1
                    dx = 1
                    dy = 0
                elif a == 2:  # left
                    x = col+1.5
                    y = row-1
                    dx = -1
                    dy = 0
                elif a == 1:  # up
                    x = col+1
                    y = row-1.5
                    dx = 0
                    dy = 1
                elif a == 3:  # down
                    x = col+1
                    y = row-0.5
                    dx = 0
                    dy = -1
                plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, 
                          head_length=0.2, color='k') 
                if counter == 24:
                    plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, 
                              head_length=0.2, color='k', label="reinforce") 
                if a_s == 0:  # right
                    x = col+0.5
                    y = row-1
                    dx = 1
                    dy = 0
                elif a_s == 2:  # left
                    x = col+1.5
                    y = row-1
                    dx = -1
                    dy = 0
                elif a_s == 1:  # up
                    x = col+1
                    y = row-1.5
                    dx = 0
                    dy = 1
                elif a_s == 3:  # down
                    x = col+1
                    y = row-0.5
                    dx = 0
                    dy = -1
                plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, 
                          head_length=0.2, color='r')
                if counter == 24:
                    plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, 
                              head_length=0.2, color='r', label ="optimal")
                counter = counter + 1
                col = col + 2
            row = row - 2
    
        ax.set_xticks(np.arange(0, 10, 2))
        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xlim([0, 10])
        plt.ylim([0,10])
        ax.grid("True")
        plt.title("Policy", fontsize = 20)
        plt.legend(bbox_to_anchor = (0.5, -0.1), loc='lower center')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        plt.close()
        
    def plot_all_policy(self, P_mat, filename):
        fig, ax = plt.subplots(1,1, figsize=(8, 8))
        row     = 10
        counter = 0
        for i in range(5):  # row
            col = 0
            for j in range(5):  # col
                
                A = P_mat[counter]
                # get coords
                for i in range(len(A)):
                    if i == 0:  # right
                        x = col+1
                        y = row-1
                        dx = 1
                        dy = 0
                    elif i == 2:  # left
                        x = col+1
                        y = row-1
                        dx = -1
                        dy = 0
                    elif i == 1:  # up
                        x = col+1
                        y = row-1
                        dx = 0
                        dy = 1
                    elif i == 3:  # down
                        x = col+1
                        y = row-1
                        dx = 0
                        dy = -1
                    
                    plt.arrow(x, y, dx*A[i]*0.7, dy*A[i]*0.7,
                              width = 0.05, head_width=0.2, 
                              head_length=0.2, color='k') 
                
                counter = counter + 1
                col = col + 2
            row = row - 2
    
        ax.set_xticks(np.arange(0, 10, 2))
        ax.set_yticks(np.arange(0, 10, 2))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xlim([0, 10])
        plt.ylim([0,10])
        ax.grid("True")
        plt.title("Policy", fontsize = 20)
        # plt.legend(bbox_to_anchor = (0.5, -0.1), loc='lower center')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        plt.close()

        
        
