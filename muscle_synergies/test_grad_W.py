import substeps
import util
import numpy as np
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time

# ==========
# This piece is for testing grad W computation
# ==========

D = 4 #number of muscles
synergy_time = 1.  #Synergy duration
T = 10  #synergy duration in time steps
N = 3 #number of synergies

#We want to construct an M that is made of shifted Gaussians added together

# construct the synergies as Gaussians
variances = np.random.uniform(1,2,(N,D))
means = np.random.uniform(3,12,(N,D))
W = util.gauss_vector(T,means,variances)


plt.ion()
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i,:].T,interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=1)
    pltuls.strip_ticks(ax)

plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

# plt.show()


#Set coeffients and use the true ones
c_min = 0
c_max = 1
true_c = np.random.uniform(c_min,c_max,N)

plt.figure(1)
for i in range(N):
    ax = plt.gcf().get_axes()[i]
    ax.text(0.5,0.5,str(
        true_c[i])[:3],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)

pre_sum_M = W*true_c[None,:,None]

print(np.shape(pre_sum_M))

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(pre_sum_M,axis=1)

S = 1
M = M.T
M = M[None,:,:]

plt.figure(200)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.plot(np.squeeze(M[:,d,:]))
plt.text(0.5,1,'Muscle Activity',transform=plt.gcf().transFigure)

true_c = true_c[None,:]

W_est = substeps.initialize_W(N,D,T) #size of W is N x D x T
# c_est = substeps.initialize_c(S,N) #size of c is S x N

plt.figure(200)
lines = []
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.ylim([0,2])
    line, = plt.plot(np.sum(W_est*np.moveaxis(true_c[:,:,None],0,2)   ,axis=0)[d,:])
    lines.append(line)

error = np.inf
t = np.zeros_like(true_c).astype(int)

error_threshold = 1e-6
error = substeps.compute_squared_error(W_est,true_c,t,M)
print('error before starting: '+str(error))

ims = []
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    im = plt.imshow(
        W_est[i,:,:],interpolation='none',
        aspect=T/D,cmap='Greys_r',vmin=0,vmax=1)
    plt.colorbar()
    ims.append(im)
    pltuls.strip_ticks(ax)
plt.show()
# raw_input(' ')
plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

while error>error_threshold:
    last = time.time()
    W_est = substeps.update_W(true_c,M,W_est,t)
    error = substeps.compute_squared_error(W_est,true_c,t,M)
    print('error after W update: '+str(error))
    for i in range(N):
    	im = ims[i]
    	im.set_data(W_est[i,:,:])
        im.set_clim(vmin=0,vmax=1)
    plt.figure(200)
    for d in range(D):
        plt.subplot(D,1,d+1)
        line = lines[d]
        line.set_ydata(np.sum(W_est*np.moveaxis(true_c[:,:,None],0,2)   ,axis=0)[d,:])

    plt.pause(0.1)
    # raw_input('next step?')





plt.show()
