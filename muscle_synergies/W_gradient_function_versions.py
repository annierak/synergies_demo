def squared_error_gradient_total_new(M,c,W,delays,i,tao):

	#Computes the gradient of (sum_s) E^2 wrt entries of w_i_tao--
	#(each dimension of w_i_tao is a muscle activity value)
	#Inputs:
	# M: S x D x T muscle activity
	# c: S x N synergy coefficents
	# W : N x D x T synergies
	# delay : S x N delays

	S,D,T = np.shape(M)
	nabla_W= np.zeros(D)
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
		nabla_W[d_0] = value
	# print(nabla_W)
	return nabla_W


def squared_error_gradient_total_old(M,c,W,delays,i,tao):

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
