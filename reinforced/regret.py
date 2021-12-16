
# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = """Prof. Carlo R. da Cunha, Ph.D. <creq@if.ufrgs.br>"""

import os

os.system('clear')

print('.-------------------------------.')
print('| REGRET Learning               |#')
print('| By.: Prof. Carlo R. da Cunha  |#')
print('|                               |#')
print('|                         2021  |#')
print('\'-------------------------------\'#')
print('  ################################')
print('')
print('Importing Libraries:')

import numpy as np
import matplotlib.pyplot as pl

# Prisoner's dillema
pmA = np.array([[2,4],[1,3]])
pmB = np.array([[2,1],[4,3]])

RA = np.array([0,0])
RB = np.array([0,0])

avA = 0
avB = 0

for it in range(100):
    # Action player A
    s = np.sum(RA)
    if (s != 0):
        p = RA[0]/s
    else:
        p = 0.5

    if (np.random.rand() < p):
        aA = 0
    else:
        aA = 1

    # Action player B    
    s = np.sum(RB)
    if (s != 0):
        p = RB[0]/s
    else:
        p = 0.5

    if (np.random.rand() < p):
        aB = 0
    else:
        aB = 1

    # Utilities for playing actions aA and aB
    uA = pmA[aA,aB]
    uB = pmB[aA,aB]

    # Maximum utilities for this round given other players moves
    maxA = np.max(pmA[:,aB])
    maxB = np.max(pmB[aA,:])

    # Regrets for playing these actions
    rA = maxA - uA
    rB = maxB - uB

    # Cumulative regrets
    RA[aA] = RA[aA] + rA
    RB[aB] = RB[aB] + rB
    
    # Average actions
    avA = (it/(it+1))*avA + aA/(it+1)
    avB = (it/(it+1))*avB + aB/(it+1)
    
    print(avA,avB)
