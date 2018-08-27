import numpy as np
import matplotlib.pyplot as plt
#utility functions for algorithm implementation and testing.

def create_ranges(start, stop, N):
    divisor = N-1
    steps = (1.0/divisor) * (stop - start)
    return steps[None,:]*np.arange(N)[:,None] + start[None,:]


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
