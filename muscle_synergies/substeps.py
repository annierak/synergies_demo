import numpy as np
import matplotlib.pyplot as plt
import flylib as flb
import time
import itertools

def initialize_W(N,D,T):
	return np.random.uniform(0,1,size=(N,D,T))

def initialize_c(S,N):
	return np.random.uniform(0,1,size=(S,N))


def compute_phi_s_i(M,W,t,i,s):
	# tao = np.range(T)
	T = np.shape(W)[2]
	summand = 0
	#want to shift W by t according to convention specified in 3.1 (i)
	W_t = shift_matrix_columns(t,W[i,:,:])
	for tao in range(T):
		summand += np.dot(M[s,:,tao],W_t[:,tao]) #synergies W are indexed by i = 1,2,...,N muscles j = 1,2,...D column is the time
	return summand

def shift_matrix_columns(column_shift,A,transpose=False):
	A_copy = np.copy(A)
	if transpose:
		A_copy = A_copy.T
	n_rows,n_cols = np.shape(A_copy)
	# if np.abs(column_shift)>n_cols:
	# 	raw_input('here'+str(column_shift))
	# column_shift = np.sign(column_shift)*(np.abs(column_shift)%n_cols)
	padding = np.zeros((n_rows,n_cols))
	double_padded = np.hstack((padding,A_copy,padding))
	starting_index = (n_cols) + (-1)*column_shift
	ending_index = starting_index+n_cols
	to_return = double_padded[:,starting_index:ending_index]
	if transpose:
		to_return = to_return.T
	return to_return


def update_delay(M,W,c,S):
	N,D,T = np.shape(W)
	delays = np.zeros((S,N)) #size of delays (t) is S x N
	for s in range(S):
		synergy_list = list(range(N))
		for synergy_step in range(N):
			phi_s_is = np.zeros((len(synergy_list),2*T+1))
			ts = range(-T,T+1)
			for i,synergy in enumerate(synergy_list):
				for delay_index,t in enumerate(ts):
					phi_s_is[i,delay_index] = compute_phi_s_i(M,W,t,synergy,s)
			# na = np.newaxis
			# ts = np.array(range(-T,T+1));ts = ts[na,:]
			# synergies = np.array(synergy_list)[na,:,na]
			# print(np.shape(W),np.shape(ts),np.shape(synergies))
			# phi_s_is = compute_phi_s_i(M,W,ts,synergies,s)

			max_synergy_index,max_delay_index = np.unravel_index(np.argmax(phi_s_is),np.shape(phi_s_is))
			max_synergy = synergy_list[max_synergy_index]
			max_delay = range(-T,T+1)[max_delay_index]
			# max_delay = range(-T,T+1)[max_delay_index]
			shifted_max_synergy = shift_matrix_columns(max_delay,W[max_synergy,:,:])
			scaled_shifted_max_synergy = c[s,max_synergy]*shifted_max_synergy
			M[s,:,:] -= scaled_shifted_max_synergy
			synergy_list.remove(max_synergy)
			delays[s,max_synergy] = max_delay
	return delays.astype(int)

def update_c(c,M,W,delays):
	S,N = np.shape(c)
	mu_c = 1e-3
	c_copy = np.copy(c)
	for s in range(S):
		delta_c = -mu_c*squared_error_gradient_episode(M[s,:,:],c[s,:],W,delays[s,:])
		c_copy[s,:] += delta_c
		print('delta_c'+str(delta_c))
	#Nonnegativity constraint
	c_copy[c_copy<0] = 0.
	return c_copy

def update_W(c,M,W,delays):
	N,D,T = np.shape(W)
	mu_W = 1e-2
	W_copy = np.copy(W)
	for i in range(N):
		for tao in range(T):
			delta_w_i_tao = -mu_W*squared_error_gradient_total(M,c,W,delays,i,tao)
			W_copy[i,:,tao] += delta_w_i_tao
	W_copy[W_copy<0] = 0
	return W_copy


def squared_error_gradient_episode(M_s,c_s,W,delays_s):
	#Computes the gradient of E_s^2 wrt entries of c_s--
	#(each dimension of c_s is a syngergy coefficent)
	#Inputs:
	# M_s: D x T muscle activity for that episode
	# c_s: N synergy coefficents
	# W : N x D x T synergies
	# delay_s : N delays
	D,T = np.shape(M_s)
	nabla = np.zeros_like(c_s)
	N = len(c_s)
	for j in range(N):
		nabla_j_ts = np.zeros(T)
		for t in range(T):
			entries_by_d = 2*(M_s[:,t]-np.sum(W[:,:,t]*c_s[:,None],axis=0))*(
				-1*(shift_matrix_columns(delays_s[j],W[j,:,:])[:,t]))
			nabla_j_ts[t] = np.sum(entries_by_d)
		nabla[j] = np.sum(nabla_j_ts)
	return nabla

def squared_error_gradient_total(M,c,W,delays,i,tao):

	#Computes the gradient of (sum_s) E^2 wrt entries of w_i_tao--
	#(each dimension of w_i_tao is a muscle activity value)
	#Inputs:
	# M: S x D x T muscle activity
	# c: S x N synergy coefficents
	# W : N x D x T synergies
	# delay : S x N delays

	S,D,T = np.shape(M)
	nabla = np.zeros(D)
	_,N = np.shape(delays)

	#The two sections immediately below should be made more efficient with object implementation

	#first, initialize a shifted W so we don't have to compute it again each time
	#produce a matrix S x N x D x T of the Ws for each s,j shift pair
	shifted_W = np.array([[shift_matrix_columns(delays[s,j],W[j,:,:]) for j in range(
		N)] for s in range(S)])
	#for every s,j pair scale the shifted W
	scaled_shifted_W = c[:,:,None,None]*shifted_W
	#then sum across each j
	summed_scaled_shifted_W = np.sum(scaled_shifted_W,axis=1)

	#make a tao/i indicator matrix of shape S x N x T
	tao_i_indicator = np.array([(tao==t-delays).astype(int) for t in range(T)]) #this could be further de-looped
	tao_i_indicator = np.moveaxis(tao_i_indicator,0,2)
	#and scale it by its coefficent in the gradient expression (c_s_j)
	scaled_tao_i_indicator = c[:,:,None]*tao_i_indicator
	#and sum accross each j
	summed_scaled_tao_i_indicator = np.sum(scaled_tao_i_indicator,axis=1)

	for d_0 in range(D):
		values_by_s_by_t = (2*M[:,d_0,:] - summed_scaled_shifted_W[:,d_0,:])*(-1)*summed_scaled_tao_i_indicator
		value = np.sum(values_by_s_by_t)
		nabla[d_0] = value
	return nabla



def compute_squared_error(W,c,t,M):
	#Returns the sum squared error across episodes as defined at the top of section 3
	S,N = np.shape(c)
	S,D,T = np.shape(M)
	error = np.zeros((S,T))
	for s in range(S):
		for t in range(T):
			entries_by_d = M[s,:,t]-np.sum(W[:,:,t]*c[s,:][:,None],axis=0)
			error[s,t] = np.sum(np.square(entries_by_d))
	return np.sum(error)
