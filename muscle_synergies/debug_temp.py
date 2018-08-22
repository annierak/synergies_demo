import numpy as np 


def shift_matrix_columns(column_shift,A,wrap_around):
	#If wrap_around = True, fills the vacated spaces with the stuff that fell off
	# otherwise fill with zeros 
	n_rows,n_cols = np.shape(A)
	A_copy = np.copy(A)
	if column_shift==0:
		return A_copy
	if wrap_around:
		return np.roll(A_copy,column_shift,axis=1)
	else:
		if column_shift>0:
			A_copy[:,column_shift:] = A[:,:-column_shift]
			A_copy[:,0:column_shift] = np.zeros((n_rows,column_shift))
		else:
			A_copy[:,:column_shift] = A_copy[:,-column_shift:]
			A_copy[:,column_shift:] = np.zeros((n_rows,-column_shift))
		return A_copy

b = np.random.randn(3,5)
print(b)
b1 = shift_matrix_columns(2,b,False)
print(b1)
b2 = shift_matrix_columns(2,b,True)
print(b2)


# print(shift_matrix_columns(-2,b,True))
