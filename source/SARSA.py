# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = """Prof. Carlo R. da Cunha, Ph.D. <creq@if.ufrgs.br>"""

import os

os.system('clear')

print('.-------------------------------.')
print('| SARSA                         |#')
print('| By.: Prof. Carlo R. da Cunha  |#')
print('|                               |#')
print('|                         2021  |#')
print('\'-------------------------------\'#')
print('  ################################')
print('')
print('Importing Libraries:')

import numpy as np
import matplotlib.pyplot as pl

#--------------#
# Dictionaries #
#--------------#

action_space = {
    "up" : 0,
    "down" : 1,
    "left" : 2,
    "right" : 3
}    

state_space = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3
}

#--------------------------#
# Epsilon-greedy algorithm #
#--------------------------#
def egreedy(state,Q):
    r1 = np.random.rand()
        
    if (r1 < 0.1):
        if (state == state_space["A"]):
            action = action_space["right"]
        elif (state == state_space["B"]):
            r2 = np.random.rand()
            if (r2 < 0.5):
                action = action_space["left"]
            else:
                action = action_space["down"]
        elif (state == state_space["C"]):
            action = action_space["right"]
        elif (state == state_space["D"]):
            r2 = np.random.rand()
            if (r2  < 0.5):
                action = action_space["left"]
            else:
                action = action_space["up"]
    else:
        action = np.argmax(Q[state,:])
        
    return action

#-----------------#
# Find next state #
#-----------------#
def next_state(state,action):
    nstate = state
    if (state == state_space["A"]):
        if (action == action_space["right"]):
            nstate = state_space["B"]
    elif (state == state_space["B"]):
        if (action == action_space["left"]):
            nstate = state_space["A"]
        elif (action == action_space["down"]):
            nstate = state_space["D"]
    elif (state == state_space["C"]):
        if (action == action_space["right"]):
            nstate = state_space["D"]
    elif (state == state_space["D"]):
        if (action == action_space["left"]):
            nstate = state_space["C"]
        elif (action == action_space["up"]):
            nstate = state_space["B"]

    return nstate

#-------------------------#
# Find reward for a state #
#-------------------------#
def reward(state):
    if (state == state_space["A"]):
        r = -1.0
    elif (state == state_space["B"]):
        r = -0.5
    elif (state == state_space["C"]):
        r = 10.0
    elif (state == state_space["D"]):
        r = -0.25
        
    return r

#===============================================#
#                     MAIN                      #
#===============================================#
alpha = 0.7
gamma = 0.95
Q = np.zeros((4,4))
y = []

for episode in range(100):
    s = state_space["C"]
    while (s == state_space["C"]):
        s = np.random.randint(4)
    a = egreedy(s,Q)

    R = 0   # Cumulated rewards
    while(s != state_space["C"]):
        r = reward(s)
        sl = next_state(s,a)
        al = egreedy(sl,Q)
                    
        Q[s][a] = (1-alpha)*Q[s][a] + alpha*(r+gamma*Q[sl][al])
        s = sl
        a = al
        R = R + r
    R = R + reward(state_space["C"])
    
    y.append(R)

# Print and plot results
print("Action-value function Q(s,a):")
print(Q)
print()

pl.plot(y)
pl.xlabel("Episode")
pl.ylabel("Cumulated Rewards")
pl.show()
