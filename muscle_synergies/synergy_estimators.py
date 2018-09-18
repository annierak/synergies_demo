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
        self.M = M #M is D x T
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

class TimeDependentSynergyEstimator(object):

    def __init__(self,S,D,T,N,M,error_threshold=1e-5):
        self.D = D
        self.T = T
        self.N = N
        self.S = S
        self.M_stacked = M #M is S x D x T
        self.M_spread = util.spread(M)
        self.SS_tot = self.compute_total_sum_squares()
        self.error_threshold = error_threshold
        self.initialization_scale = 1.

        self.check_shapes()
        self.W_est_stacked = np.random.uniform(
            0,self.initialization_scale,size=(self.N,self.D,self.T))
        self.W_est_spread = util.spread(self.W_est_stacked)
        self.c_est = np.random.uniform(
            0,self.initialization_scale,size=(self.S,self.N))
        self.delays = np.zeros_like(self.c_est) #******** change this

        self.M_est_stacked = np.sum(self.W_est_stacked[None,:,:,:]*self.c_est[:,:,None,None],axis=1)
        self.M_est_spread = util.spread(self.M_est_stacked)

        self.construct_Theta()

    def check_shapes(self): #******** this should be more thorough
        if np.shape(self.M_stacked) != (self.S,self.D,self.T):
            raise Exception('M is not SxDxT, it is shape '+str(np.shape(self.M)))

    def compute_total_sum_squares(self):
        return np.sum(np.square(self.M_stacked-np.mean(self.M_stacked)))

    def compute_squared_error(self):
        #Returns the sum squared error across episodes as defined at the top of section 3
        #Aka residual sum of squares
        #This needs to be phased out--switch to the trace version with M-WH
        error = np.zeros((self.S,self.T))
        for s in range(self.S):
            for t in range(self.T):
                entries_by_d = self.M_stacked[s,:,t]-np.sum(
                    self.W_est_stacked[:,:,t]*self.c_est[s,:][:,None],axis=0)
                error[s,t] = np.sum(np.square(entries_by_d))
        return np.sum(error)

    def compute_R2(self):
        SS_res = self.compute_squared_error() #*****change this to have actual shifts
        return  (1. - (SS_res/self.SS_tot))

    def update_M_est(self): #********** this doesn't have t capacities yet
        self.M_est_stacked = np.sum(self.W_est_stacked[None,:,:,:]*self.c_est[:,:,None,None],axis=1)
        self.M_est_spread = util.spread(self.M_est_stacked)

    def compute_phi_s_i(self,M_temp,t,i,s):
        summand = 0
        #want to shift W by t according to convention specified in 3.1 (i)
        W_t = util.shift_matrix_columns(t,self.W_est_stacked[i,:,:])
        for tao in range(self.T):
            summand += np.dot(M_temp[s,:,tao],W_t[:,tao]) #synergies W are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time
        return summand

    def update_delays(self):
        for s in range(self.S):
            M_copy = np.copy(self.M_stacked)
            synergy_list = list(range(self.N))
            for synergy_step in range(self.N):
                phi_s_is = np.zeros((len(synergy_list),2*self.T+1))
                ts = range(-self.T,self.T+1)
                for i,synergy in enumerate(synergy_list):
                    for delay_index,t in enumerate(ts):
                        phi_s_is[i,delay_index] = self.compute_phi_s_i(
                            M_copy,t,synergy,s)
                max_synergy_index,max_delay_index = np.unravel_index(
                    np.argmax(phi_s_is),np.shape(phi_s_is))
                max_synergy = synergy_list[max_synergy_index]
                max_delay = range(-self.T,self.T+1)[max_delay_index]
                shifted_max_synergy = util.shift_matrix_columns(
                    max_delay,self.W_est_stacked[max_synergy,:,:])
                #Original scaling attempt
                scaled_shifted_max_synergy = self.c_est[s,max_synergy]*shifted_max_synergy
                M_copy[s,:,:] -= scaled_shifted_max_synergy
                #This is the piece where we assume we make M nonnegative
                M_copy[M_copy<0] =0.
                synergy_list.remove(max_synergy)
                self.delays[s,max_synergy] = int(max_delay)

    def construct_Theta(self):
        N = self.N
        T = self.T
        Theta = np.zeros((N,2*T-1,N*T,T)) #shape of each Theta_i(t) is N*T x T
        for i in range(1,N+1):
            for t in range(1-T,T):
                rows,columns = np.indices((N*T,T))
                to_fill = (rows+1-(i-1)*T)==(columns+1-t)
                to_fill[0:(i-1)*T,:] = 0.
                to_fill[i*T:,:] = 0.
                Theta[i-1,util.t_shift_to_index(t,T),:,:] = to_fill
        self.Theta = Theta

    def update_H(self):
        H = np.zeros((self.S,self.N*self.T,self.T))
        for s in range(self.S):
            H[s,:,:] = np.sum(self.c_est[s,:][:,None,None]*\
                np.array([self.Theta[i,util.t_shift_to_index(
                self.delays[s,i], self.T),:,:] for i in range(self.N)]),axis=0)
        self.H_stacked = H
        self.H_spread = np.concatenate(H,axis=1)

    def update_c_est(self,scale=1):
        mult_factor = np.zeros_like(self.c_est)
        for s in range(self.S):
            for i in range(self.N):
                Theta_i_tis = self.Theta[i,util.t_shift_to_index(self.delays[s,i],self.T),:,:]
                num = np.trace((self.M_stacked[s,:,:].T).dot(self.W_est_spread).dot(Theta_i_tis))
                denom = np.trace((self.H_stacked[s,:,:].T).dot(
                    self.W_est_spread.T).dot(self.W_est_spread).dot(Theta_i_tis))
                mult_factor[s,i] = num/denom
        self.c_est = self.c_est*scale*mult_factor

    def update_W_est(self,scale=1):
        zeros = (self.W_est_spread.dot(self.H_spread).dot(self.H_spread.T)==0)
        nonzero_indices = np.logical_not(zeros)
        mult_factor = scale*np.dot(self.M_spread,self.H_spread.T)/(
            self.W_est_spread.dot(self.H_spread).dot(self.H_spread.T))
        self.W_est_spread[nonzero_indices] = self.W_est_spread[
            nonzero_indices]*mult_factor[nonzero_indices]
        self.W_est_stacked = util.stack(self.W_est_spread,self.T)
