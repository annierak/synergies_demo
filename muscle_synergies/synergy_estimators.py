#Object-based implementation of time-dependent and synchronous synergies

import substeps
import util
import numpy as np
import matplotlib
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys
import scipy.linalg


class SynchronousSynergyEstimator(object):
    def __init__(self,D,T,N,M,error_threshold=5e-4):
        self.D = D
        self.T = T
        self.N = N
        self.M = M #M is D x N
        self.SS_tot = np.sum(np.square(M-np.mean(M)))
        self.error_threshold = error_threshold

        self.check_shapes()

        self.C_est = np.random.uniform(0,1,(N,T))
        self.W_est = np.random.uniform(0,1,(D,N))
        self.M_est = np.dot(self.W_est,self.C_est)

    def check_shapes(self):
        if np.shape(self.M) != (self.D,self.T):
            raise Exception('M is not DxT, it is shape '+str(np.shape(self.M)))

    def compute_error_by_trace(self):
        diff = self.M-(self.W_est).dot(self.C_est)
        return np.trace(np.dot(diff.T,diff))

    def compute_oneminusrsq(self):
        SS_res = self.compute_error_by_trace()
        return SS_res/self.SS_tot

    def update_C(self):
        num = np.dot(self.W_est.T,self.M)
        denom = (self.W_est.T).dot(self.W_est).dot(self.C_est)
        self.C_est = self.C_est*(num/denom)

    def update_W(self):
        mult_factor = np.dot(self.M,self.C_est.T)/(
            self.W_est.dot(self.C_est).dot(self.C_est.T))
        self.W_est = self.W_est*mult_factor
