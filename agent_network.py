#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 13:30:48 2023

@author: john
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

class PGN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGN, self).__init__()
        self.fc1      = nn.Linear(input_size, hidden_size)
        # self.dropout1 = nn.Dropout(p = 0.4)
        self.fc2      = nn.Linear(hidden_size, output_size)
        # self.dropout2 = nn.Dropout(p=0.4)
        # self.fc3      = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        # x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = F.softmax(self.fc2(x))
        return x

class Agent:
    def __init__(self, env, hidden_size, learning_rate,
                 gamma, device, opt_type):
        self.n_observations = 1
        self.n_actions      = env.num_actions
        self.hidden_size    = hidden_size
        self.learning_rate  = learning_rate
        self.gamma          = gamma
        self.device         = device
        self.env            = env
        
        
        self.policy_network = PGN(self.n_observations, hidden_size,
                                  self.n_actions).to(device)
        if opt_type == "adam":
            self.optimizer  = optim.Adam(self.policy_network.parameters(), 
                                        lr = learning_rate)
        elif opt_type == "sgd":
            self.optimizer  = optim.SGD(self.policy_network.parameters(), 
                                        lr = learning_rate)

    def act(self, state):
        state  = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs  = self.policy_network(state).cpu()
        m      = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def reinforce(self, n_training_episodes, print_every, weight_dir):
        scores_deque = deque(maxlen=100)
        scores       = []
        all_rewards  = []
        max_mean     = 0
        for i_episode in range(1, n_training_episodes + 1):
            saved_log_probs = []
            rewards = []
            state = np.array(self.env.reset())
            done  = False
            while not done:
                action, log_prob = self.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done = self.env.step(action)
                state = np.array(state)
                rewards.append(reward)
            all_rewards.append(rewards)
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
    
           
            returns = deque(maxlen=self.env.max_num_steps)
            n_steps = len(rewards)
           
            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(self.gamma * disc_return_t + rewards[t])
    
            
            eps     = np.finfo(np.float32).eps.item()
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
    
           
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.stack(policy_loss, dim = 0).sum()
    
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            mean_scores = np.mean(scores_deque)
            if mean_scores > max_mean:
                max_mean = mean_scores
                torch.save(self.policy_network.state_dict(),
                           weight_dir + 'pnet_model_weights_max_mean.pth')
    
            if i_episode % print_every == 0:
                print("Episode {}\t Total reward: {}\tAverage Score: {:.2f}"
                      .format(i_episode, scores[-1], np.mean(scores_deque)))

        return all_rewards
    
def TD0(env, test_net, alpha, gamma, total_episodes, device):
    V = np.zeros(env.num_states)
    for eps in range(total_episodes):
        s = np.array(env.reset())
        done = False
        while not done:
            s      = torch.from_numpy(s).float().unsqueeze(0).to(device)
            probs  = test_net(s).cpu()
            m      = Categorical(probs)
            action = m.sample()
            s1, reward, done = env.step(action.item())
            target  = reward + gamma*V[int(s1)]
            V[int(s[0])] += alpha * (target - V[int(s[0])])
            s       = np.array(s1)
            
    return V


def initialize(env):
    V      = np.zeros(env.num_states)
    policy = np.random.randint(env.num_actions, size=env.num_states)
    return V, policy

def value_iteration(env, gamma, tol, max_iterations, Vl):
    V, policy    = initialize(env)
    A            = np.arange(env.num_actions)
    for i in range(max_iterations):
        delta    = 0
        for s in range(env.num_states):
            Q_a         = np.zeros(env.num_actions)
            for a in A:
                for s1 in range(env.num_states):
                    Q_a[a]      += env.p(s1, s, a)* \
                             (env.r(s,a) + gamma*V[s1])
            
            v           = np.max(Q_a)
            policy[s]   = np.argmax(Q_a)
            delta       = max(delta, np.abs(v - V[s]))
            V[s]        = v
        Vl.append(np.mean(V))
        if delta < tol: 
            break   
    return V, policy 