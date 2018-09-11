import substeps
import util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys


D = 5 #number of muscles
S = 50 #number of episodes
T = 15    #time samples
N = 3 #number of synergies

#Generate coeffients

C = np.random.uniform(0,1,(N,T))
W = np.random.uniform(0,1,(D,N))

M = np.dot(W,C)

plt.figure(1)

for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i].T[:,None],interpolation='none',
    aspect=3./D,cmap='YlGnBu',vmin=0,vmax=1)
    pltuls.strip_bare(ax,axis='x')
    pltuls.strip_ticks(ax,axis='y')


plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

plt.show()
