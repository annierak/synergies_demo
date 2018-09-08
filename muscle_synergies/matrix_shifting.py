import numpy as np
import matplotlib.pyplot as plt
import util
import plotting_utls as pltuls

T = 20
D = 5
N = 1

#Make a W
variances = np.random.uniform(0.5*T/20,T/20,(N,D))
means = np.random.uniform(1,T,(N,D))
W = util.gauss_vector(T,means,variances)

amp_max = 1
amplitudes = np.random.uniform(0,amp_max,(1,N,D))
c_min = 0
c_max = 1

W = amplitudes*W
W = np.squeeze(np.moveaxis(W,0,2))

shifts = np.array([-1,2,4])
shift_indices = shifts + (T-1)

plt.subplot2grid((len(shifts)+1,2),(0,1))
plt.imshow(W,interpolation='none')

# padded_W = np.concatenate([np.zeros((D,T-1)),W,np.zeros((D,T))],axis=1)

counter = 0
shift_matrices = []
for shift,index in list(zip(shifts,shift_indices)):
    shift_matrix = np.zeros((T,3*T-1))
    print(np.shape(shift_matrix))
    shift_matrix[:,index:index+T] = np.identity(T)
    plt.subplot2grid((len(shifts)+1,2),(counter+1,0))
    plt.imshow(shift_matrix,interpolation='none')
    shift_matrix = shift_matrix[:,T-1:2*T]
    shift_matrices.append(shift_matrix)
    shifted_W = W.dot(shift_matrix)
    ax = plt.subplot2grid((len(shifts)+1,2),(counter+1,1))
    plt.imshow(shifted_W,interpolation='none')
    pltuls.strip_bare(ax)
    plt.ylabel(shift,rotation=0)

    counter +=1

plt.show()

big_shift_matrix = np.vstack(shift_matrices)
