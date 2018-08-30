import substeps
import util
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure()
for i in range(N):
    plt.subplot(N,1,i+1)
    plt.imshow(W[:,i,:].T,interpolation='none',aspect=T/D,cmap='Greys_r')
    # for j in range(D):
    #     plt.subplot(D,1,j+1)
    #     plt.plot(W[:,i,j],color = 'blue')
    #     plt.ylim([0,amp_max*c_max])

plt.show()


#Make some coeffients c and multiply W by that coefficient for each synergy


true_c = np.random.uniform(c_min,c_max,N)

pre_sum_M = W*true_c[None,:,None]

for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(pre_sum_M[:,i,j],color = 'red')

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(pre_sum_M,axis=1)

plt.figure(N+1)
for j in range(D):
    plt.subplot(D,1,j+1)
    plt.plot(M[:,j],color = 'purple')

W_est = substeps.initialize_W(N,D,T) #size of W is N x D x T
c_est = substeps.initialize_c(S,N) #size of c is S x N

error = np.inf

error_threshold = 1e-6

while error>error_threshold:
	last = time.time()
	delays = substeps.update_delay(M,W_est,c_est,S) #size of delays (t) is S x N
	error = substeps.compute_squared_error(W,c,t,M)

	c_est = substeps.update_c(c_est,M,W_est,delays)
	error = substeps.compute_squared_error(W,c,t,M)

	W_est = substeps.update_W(c_est,M,W_est,delays)
	error = substeps.compute_squared_error(W,c,t,M)
	print(time.time()-last)



# plt.show()