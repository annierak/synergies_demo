import substeps
import util
import numpy as np
import matplotlib.pyplot as plt

# ==========
# This piece is for testing step 2 of the algorithm (delay selection)
# ==========

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
# shifts = np.floor(np.random.uniform(-100,100,N)).astype(int)
shifts = np.zeros(N).astype(int)

print('shifts: '+str(shifts))

shifted_W = np.zeros_like(W)

for i in range(N):
    shifted_W[:,i,:] = substeps.shift_matrix_columns(shifts[i],W[:,i,:],transpose=True)

#Make some coeffients c and multiply W by that coefficient for each synergy

c_min = 0
c_max = 4

true_c = np.random.uniform(c_min,c_max,N)
print('true c: '+str(true_c))

scaled_shifted_W = shifted_W*true_c[None,:,None]

for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(scaled_shifted_W[:,i,j],color = 'purple')

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(scaled_shifted_W,axis=1)

#Now plot M to demonstrate it is a sum of the shifted W
plt.figure(N+1)
for j in range(D):
    plt.subplot(D,1,j+1)
    plt.plot(M[:,j],color = 'purple')

#Now, we input the M and ORIGINAL W
#AS WELL AS randomly initialized cs and ts (ts small or zero)
#into the coefficent update function to test
#whether the initial cs are updated to get closer to the true cs

S = 1
c = np.random.uniform(c_min,c_max,(S,N))
print('initial c guess:'+str(c))
# delays = np.floor(np.random.uniform(0,5,(S,N))).astype(int)
delays = np.zeros((S,N)).astype(int)
M = M.T
M = M[None,:,:]
W = np.moveaxis(W,0,2)

updated_c = substeps.update_c(c,M,W,delays)
print('updated c estimate: '+str(updated_c))


#Check that the compute_squared_error is working using true cs:

print(substeps.compute_squared_error(W,true_c[None,:],delays,M))

error_initial = substeps.compute_squared_error(W,c,delays,M)
error_updated = substeps.compute_squared_error(W,updated_c,delays,M)

print('error of initial estimate: '+str(error_initial))
print('error of updated estimate: '+str(error_updated))

plt.show()
