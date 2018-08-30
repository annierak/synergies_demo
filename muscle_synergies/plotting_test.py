import substeps
import util
import numpy as np
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time

N = 3
D = 5
T = 15

plt.ion()
plt.figure(1)
# for i in range(N):
#     ax = plt.subplot2grid((N,2),(i,0))
#     plt.imshow(np.random.randn(D,T),interpolation='none',aspect=T/D,cmap='Greys_r')
#     pltuls.strip_ticks(ax)
#
# plt.text(0.3,0.95,'True W',transform=plt.gcf().transFigure)

#
ims = []
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    im = ax.imshow(np.random.randn(D,T),interpolation='none',aspect=T/D,cmap='Greys_r')
    ims.append(im)
    pltuls.strip_ticks(ax)
# plt.show()


for j in range(100):
    for i in range(N):
        im = ims[i]
        test_im = np.random.randn(D,T)
        im.set_array(test_im)
    plt.draw()
    plt.pause(0.1)
