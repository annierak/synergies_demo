import numpy as np

def shift_matrix_columns(column_shift,A):
	n_rows,n_cols = np.shape(A)
	column_shift = np.sign(column_shift)*(np.abs(column_shift)%n_cols)
	padding = np.zeros((n_rows,n_cols))
	double_padded = np.hstack((padding,A,padding))
	starting_index = (n_cols) + (-1)*column_shift
	ending_index = starting_index+n_cols
	return double_padded[:,starting_index:ending_index]

b = np.random.randn(3,5)
print(b)
# b1 = shift_matrix_columns(2,b,False)
# print(b1)
# b2 = shift_matrix_columns(2,b,True)
# print(b2)

b2 = shift_matrix_columns(-1,b)
print(b2)

b2 = shift_matrix_columns(-6,b)
print(b2)
