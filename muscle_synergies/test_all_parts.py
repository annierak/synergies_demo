import substeps
import util
import numpy as np
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time

# ==========
# This piece is for testing the entire updating process (shifts, coefficients, Ws)
# ==========

D = 5 #number of muscles
synergy_time = 1.  #Synergy duration
S = 1
T = 15  #synergy duration in time steps
N = 3 #number of synergies

#We want to construct an M that is made of shifted Gaussians added together

# construct the synergies as Gaussians
variances = np.random.uniform(0.5*T/20,T/20,(N,D))
means = np.random.uniform(1,T,(N,D))
W = util.gauss_vector(T,means,variances)


#Scale the synergies with a different amplitude for each muscle/synergy time course
amp_max = 1
amplitudes = np.random.uniform(0,amp_max,(1,N,D))
c_min = 0
c_max = 1

W = amplitudes*W

plt.ion()
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i,:].T,interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    pltuls.strip_ticks(ax)

plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

# plt.show()


#Make some coeffients c and multiply W by that coefficient for each synergy

display_episode = 0

true_c = np.random.uniform(c_min,c_max,(S,N))
plt.figure(1)
for i in range(N):
    ax = plt.gcf().get_axes()[i]
    ax.text(0.5,0.5,str(
        true_c[display_episode,i])[:3],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
W = np.moveaxis(W,0,2)


pre_sum_M = W[None,:,:,:]*true_c[:,:,None,None]



#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(pre_sum_M,axis=1)

W_est = substeps.initialize_W(N,D,T) #size of W is N x D x T
c_est = substeps.initialize_c(S,N) #size of c is S x N

plt.figure(200)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.plot(M[display_episode,d,:])
plt.text(0.5,1,'Muscle Activity',transform=plt.gcf().transFigure)
lines = []
M_est = np.sum(W_est[None,:,:,:]*c_est[:,:,None,None],axis=1)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.ylim([0,1])
    print(len(M_est[display_episode,d,:]))
    line, = plt.plot(M_est[display_episode,d,:])
    lines.append(line)

plt.figure(300)
im = plt.imshow(
    M[0,:,:],interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
pltuls.strip_ticks(ax)
plt.text(0.65,0.95,'True M',transform=plt.gcf().transFigure)


error = np.inf

error_threshold = 1e-6
error = substeps.compute_squared_error(W_est,c_est,np.zeros_like(c_est),M)
print('error before starting: '+str(error))

ims = []
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    im = plt.imshow(
        W_est[i,:,:],interpolation='none',
        aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    plt.colorbar()
    ims.append(im)
    pltuls.strip_ticks(ax)
# plt.show()

plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

counter = 1
while error>error_threshold:
    last = time.time()
    t = substeps.update_delay(M,W_est,c_est,S) #size of delays (t) is S x N
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('error after delay update: '+str(error))
    c_est = substeps.update_c(c_est,M,W_est,t)
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('c_est: ' + str(c_est[display_episode]))
    print('true_c: ' + str(true_c[display_episode]))
    print('error after c update: '+str(error))
    W_est = substeps.update_W(c_est,M,W_est,t)
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('error after W update: '+str(error))
    print('delays: '+str(t))
    for i in range(N):
    	im = ims[i]
    	im.set_data(W_est[i,:,:])
        im.set_clim(vmin=0,vmax=amp_max)
    plt.figure(200)
    W_est_shift = np.array([substeps.shift_matrix_columns(int(t[display_episode,i]),W_est[i,:,:]) for i in range(N)])
    M_est = np.squeeze(np.sum(W_est_shift[None,:,:,:]*c_est[display_episode,:,None,None],axis=1))
    for d in range(D):
        plt.subplot(D,1,d+1)
        line = lines[d]
        line.set_ydata(M_est[d,:])
    plt.draw()
    plt.pause(0.02)
    print('ITERATION: '+str(counter))
    counter+=1




plt.show()
