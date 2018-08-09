import numpy as np
import matplotlib.pyplot as plt
import flylib as flb
import time
import itertools

def initialize_W(N,D,T):
	return np.random.uniform(0,1,size=(N,D,T))

def initialize_c(N):
	return np.random.uniform(0,1,size=(N,D))


def compute_phi_s_i(M,W,t,i):
	# tao = np.range(T)
	summand = 0
	for tao in range(T):
		summand += np.dot(M[tao,:],W[i,j,tao-t]) #synergies are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time
	return summand

def shift_matrix_columns(column_shift,A):
	n_rows,_ = np.shape(A)
	if column_shift==0:
		return A
	if column_shift>0:
		A[column_shift:] = A[:-column_shift]
		A[0:column_shift] = np.zeros(n_rows,column_shift)
	else:
		A[:-column_shift] = A[column_shift:]
		A[-column_shift:] = np.zeros(n_rows,column_shift)
def update_delay(M,W,c,T):
	N,D,T = np.shape(W)
	synergy_list = list(range(N))
	delays = np.zeros(N)
	for synergy_step in range(N)
		phi_s_is = np.zeros(N,t)
		for i in synergy_list:
			for t in range(T):
				phi_s_is[i,t] = compute_phi_s_i(M,W,t,i)
		max_synergy,max_delay = np.unravel_index(np.argmax(phi_s_is),np.shape(phi_s_is))
		shifted_max_synergy = shift_matrix_columns(max_delay,W[max_synergy,:,:])
		scaled_shifted_max_synergy = c[max_synergy]*shifted_max_synergy
		adjusted_M = M - scaled_shifted_max_synergy
		synergy_list.remove(max_synergy)
		delays[synergy_step] = max_delay
	return(delays)







