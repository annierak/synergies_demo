import substeps
import util
import numpy as np
import matplotlib.pyplot as plt

# ==========
# This piece is for testing step 1 of the algorithm (delay selection)
# ==========

S = 4 #number of episodes, indexed by s
D = 5 #number of muscles
synergy_time = 1.  #Synergy duration
T = 500  #synergy duration in time steps
N = 3 #number of synergies

#We want to construct an M that is made of shifted Gaussians added together

# construct the synergies as Gaussians
variances = np.random.uniform(10,20,(N,D))
means = np.random.uniform(100,400,(N,D))
W = util.gauss_vector(T,means,variances)

#display each synergy in its own plot
for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(W[:,i,j],color='red')
# plt.show()

#now shift the synergies, and check to make sure they're shifted by plotting
#(this also tests the matrix_shift function)
shifts = np.floor(np.random.uniform(-100,100,N)).astype(int)

print(shifts)

shifted_W = np.zeros_like(W)

for i in range(N):
    shifted_W[:,i,:] = substeps.shift_matrix_columns(shifts[i],W[:,i,:],transpose=True)
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(shifted_W[:,i,j],color = 'purple')

#Now make M by adding each of these shifted N synergies muscle-wise
M = np.sum(shifted_W,axis=1)

#Now plot M to demonstrate it is a sum of the shifted W
plt.figure(N+1)
for j in range(D):
    plt.subplot(D,1,j+1)
    plt.plot(M[:,j],color = 'purple')


#Now, we input the M and ORIGINAL W into the delay selection function to test
#whether we can recover the shifts
S = 1
c = np.ones((S,N))
M = M.T
M = M[None,:,:]
W = np.moveaxis(W,0,2)
delays = substeps.update_delay(M,W,c,S)

print(delays)
plt.show()
