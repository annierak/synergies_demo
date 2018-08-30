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
T = 15  #synergy duration in time steps
N = 3 #number of synergies

#We want to construct an M that is made of shifted Gaussians added together

# construct the synergies as Gaussians
variances = np.random.uniform(1,5,(N,D))
means = np.random.uniform(3,12,(N,D))
W = util.gauss_vector(T,means,variances)


#Scale the synergies with a different amplitude for each muscle/synergy time course
amp_max = 4
amplitudes = np.random.uniform(1,amp_max,(1,N,D))
c_min = 0
c_max = 4

W = amplitudes*W

plt.ion()
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i,:].T,interpolation='none',aspect=T/D,cmap='Greys_r')
    pltuls.strip_ticks(ax)

plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

# plt.show()


#Make some coeffients c and multiply W by that coefficient for each synergy


true_c = np.random.uniform(c_min,c_max,N)

pre_sum_M = W*true_c[None,:,None]

# for i in range(N):
#     plt.figure(i)
#     for j in range(D):
#         plt.subplot(D,1,j+1)
#         plt.plot(pre_sum_M[:,i,j],color = 'red')

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(pre_sum_M,axis=1)

# plt.figure(N+1)
# for j in range(D):
#     plt.subplot(D,1,j+1)
#     plt.plot(M[:,j],color = 'purple')

S = 1
M = M.T
M = M[None,:,:]


true_c = true_c[None,:]

W_est = substeps.initialize_W(N,D,T) #size of W is N x D x T
c_est = substeps.initialize_c(S,N) #size of c is S x N

error = np.inf

error_threshold = 1e-6
error = substeps.compute_squared_error(W_est,c_est,np.zeros_like(c_est),M)
print('error before starting: '+str(error))

ims = []
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    im = plt.imshow(W_est[i,:,:],interpolation='none',aspect=T/D,cmap='Greys_r')
    ims.append(im)
    pltuls.strip_ticks(ax)
# plt.show()

plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

while error>error_threshold:
    last = time.time()
    t = substeps.update_delay(M,W_est,c_est,S) #size of delays (t) is S x N
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('error after delay update: '+str(error))
    c_est = substeps.update_c(c_est,M,W_est,t)
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('error after c update: '+str(error))
    W_est = substeps.update_W(c_est,M,W_est,t)
    error = substeps.compute_squared_error(W_est,c_est,t,M)
    print('error after W update; '+str(error))
    for i in range(N):
    	im = ims[i]
    	im.set_data(W_est[i,:,:])
    plt.draw()
    time.sleep(1)
    plt.pause(1)





plt.show()
