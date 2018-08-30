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

	for d_0 in range(D):
		values_by_s_by_t = np.zeros((S,T))
		for t in range(T):
			for s in range(S):
				# second_term,third_term = np.zeros(N),np.zeros(N)
				second_term = np.array([c[s,j]*shift_matrix_columns(delays[s,j],W[j,:,:])[d_0,t] for j in range(N)])
				third_term = np.array([c[s,j]*(tao==t-delays[s,j]) for j in range(N)])
				entry = (2*M[s,d_0,t] - np.sum(second_term))*(-1)*np.sum(third_term)
				values_by_s_by_t[s,t] = entry
		value = np.sum(values_by_s_by_t)
		nabla[d_0] = value
	return nabla
