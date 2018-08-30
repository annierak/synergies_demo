import substeps
import util
import numpy as np
import matplotlib.pyplot as plt

# ==========
# This piece is for testing step 3 of the algorithm (W update)
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


#now shift the synergies, and check to make sure they're shifted by plotting
#(this also tests the matrix_shift function)
# shifts = np.floor(np.random.uniform(-100,100,N)).astype(int)
shifts = np.zeros(N).astype(int)


shifted_W = np.zeros_like(W)

for i in range(N):
    shifted_W[:,i,:] = substeps.shift_matrix_columns(shifts[i],W[:,i,:],transpose=True)

#Make some coeffients c and multiply W by that coefficient for each synergy

c_min = 0
c_max = 4

true_c = np.random.uniform(c_min,c_max,N)

scaled_shifted_W = shifted_W*true_c[None,:,None]

for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(scaled_shifted_W[:,i,j],color = 'red')

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.sum(scaled_shifted_W,axis=1)

#Now plot M to demonstrate it is a sum of the shifted W
plt.figure(N+1)
for j in range(D):
    plt.subplot(D,1,j+1)
    plt.plot(M[:,j],color = 'purple')

#Now, we input the M, the CORRECT cs and ts (since this part is not updating those)
# and a slightly warped W
# into the W update function to test
# whether the W values are updated to get closer to the true W values
# the way we'll make the "W guess" is to preserve the proper cs and cs, but just
# add some positive noise to the true value at a few time chunks
# so that the heights are a bit messed up

noise_scale = 0.2
perturbed_indices = np.zeros_like(scaled_shifted_W)
print(np.shape(perturbed_indices))
perturbed_indices[100:150,1,2] = 1
perturbed_indices[300:350,2,1] = 1
perturbed_indices[200:250,0,4] = 1
W_inital = scaled_shifted_W+np.random.uniform(0,noise_scale,np.shape(scaled_shifted_W))*perturbed_indices
W_inital[W_inital<0]=0
#Put the W_initial on top of the true W plots
for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(W_inital[:,i,j],color = 'blue')
# plt.show()
S = 1
M = M.T
M = M[None,:,:]
W_inital = np.moveaxis(W_inital,0,2)

shifts = shifts[None,:]
true_c = true_c[None,:]

error_initial = substeps.compute_squared_error(W_inital,true_c,shifts,M)
print('error of initial estimate: '+str(error_initial))

for update in range(10):
    W_updated = substeps.update_W(true_c,M,W_inital,shifts)
    error_updated = substeps.compute_squared_error(W_updated,true_c,shifts,M)
    print('error of updated estimate: '+str(error_updated))
    W_inital = W_updated


#Put the updated W on the plots with initial and true W
for i in range(N):
    plt.figure(i)
    for j in range(D):
        plt.subplot(D,1,j+1)
        plt.plot(W_updated[i,j,:],color = 'purple')


error_baseline = substeps.compute_squared_error(np.moveaxis(W,0,2),true_c,shifts,M)
print('error of true: '+str(error_baseline))

plt.show()
