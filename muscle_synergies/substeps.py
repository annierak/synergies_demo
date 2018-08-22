import numpy as np
import matplotlib.pyplot as plt
import flylib as flb
import time
import itertools

def initialize_W(N,D,T):
	return np.random.uniform(0,1,size=(N,D,T))

def initialize_c(N,D):
	return np.random.uniform(0,1,size=(N,D))


def compute_phi_s_i(M,W,t,i):
	# tao = np.range(T)
	T = np.shape(W)[2]
	summand = 0
	#want to shift W by -t according to convention specified in 3.1 (i)
	W_t = shift_matrix_columns(-t,W[i,:,:])
	for tao in range(T):
		summand += np.dot(M[:,tao],W_t[:,tao]) #synergies W are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time
	return summand

def shift_matrix_columns(column_shift,A,wrap_around=False):
	#If wrap_around = True, fills the vacated spaces with the stuff that fell off
	# otherwise fill with zeros 
	n_rows,n_cols = np.shape(A)
	if column_shift>n_cols:
		column_shift = column_shift % n_cols
	# print(n_rows,n_cols)
	A_copy = np.copy(A)
	if column_shift==0:
		return A_copy
	if wrap_around:
		return np.roll(A_copy,column_shift,axis=1)
	else:
		if column_shift>0:
			A_copy[:,column_shift:] = A_copy[:,:-column_shift]
			A_copy[:,:column_shift] = np.zeros((n_rows,column_shift))
		else:
			A_copy[:,:column_shift] = A_copy[:,-column_shift:]
			A_copy[:,column_shift:] = np.zeros((n_rows,-column_shift))
		return A_copy

def update_delay(M,W,c):
	N,D,T = np.shape(W)
	synergy_list = list(range(N))
	delays = np.zeros(N)
	for synergy_step in range(N):
		phi_s_is = np.zeros((N,2*T+1))
		for i in synergy_list:
			print(i)
			for t in range(-T,T+1):
				phi_s_is[i,t-1] = compute_phi_s_i(M,W,t,i)
		max_synergy,max_delay = np.unravel_index(np.argmax(phi_s_is),np.shape(phi_s_is))
		shifted_max_synergy = shift_matrix_columns(max_delay,W[max_synergy,:,:])
		scaled_shifted_max_synergy = np.multiply(c[max_synergy],shifted_max_synergy.T).T
		M -= scaled_shifted_max_synergy
		synergy_list.remove(max_synergy)
		delays[max_synergy] = max_delay
	return(delays)







