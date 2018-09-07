import numpy as np
import matplotlib.pyplot as plt
#utility functions for algorithm implementation and testing.

def create_ranges(start, stop, N):
    divisor = N-1
    steps = (1.0/divisor) * (stop - start)
    return steps[None,:]*np.arange(N)[:,None] + start[None,:]


def spread(A):
    #Turns stacked matrix into adjacent matrix spread for multiplicative update
    #reshape from d x e x T to  e x (dxT)
    B = np.copy(A)
    return np.concatenate(B,axis=1)


def stack(A,T):
    #Inverse of above--turns shape of A from e x d*T to d x e x T
    e,dT = np.shape(A)
    B = np.copy(A)
    return np.reshape(B,(-1,e,T))


def t_index_to_shift(index,T):
	#for a given time duration T, switch the index of range(1-T,T) to the shift value
	return (1-T)+index
def t_shift_to_index(shift,T):
	#do the reverse of the above
	return shift +(T-1)

def construct_H(c_est,Theta,delays):
    S,N = np.shape(delays)
    _,_,_,T = np.shape(Theta)
    H = np.zeros((S,N*T,T))
    for s in range(S):
        # print(np.shape(c_est[s,:][:,None,None]))
        # print(np.shape(np.array([Theta[i,t_shift_to_index(delays[s,i],T),:,:] for i in range(N)])))

        H[s,:,:] = np.sum(c_est[s,:][:,None,None]*\
            np.array([Theta[i,t_shift_to_index(
            delays[s,i],T),:,:] for i in range(N)]),axis=0)
    #last bit is to turn H from S x N*T x T to N*T x T*S
    return np.concatenate(H,axis=1)


def gauss_vector(max_x,mu,sigma):
    #Returns a vector of shape (max_x, shape(mu)=shape(sigma)) whose values are a Gaussian
    #function centered on mu, std sigma
    inputs = create_ranges(np.array([1]),max_x,max_x)
    if len(np.shape(mu))>1:
        inputs = inputs[:,:,None]
    return np.exp(-np.power(inputs - mu[None,:], 2.) / (2 * np.power(sigma[None,:], 2.)))

# b = np.zeros((4,100))
def gauss_vector_demo():
    mus = np.array([[10,20],[10,20],[10,20]])
    sigma = np.array([[10,10],[20,20],[30,30]])
    # mus = np.array([10,20,30])
    # sigma = np.array([10,10,10])
    max_x = 100
    output = gauss_vector(max_x,mus,sigma)
    plt.plot(np.reshape(output,(100,6)))
    # to_plot = gauss_vector(100,np.array([50]),np.array([10]))
    # plt.plot(to_plot)
    plt.show()

# gauss_vector_demo()
