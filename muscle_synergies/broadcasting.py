import numpy as np 

def my_func(a,b):
	if a>b:
		return np.cos(a-b)/(b**2)
	else:
		return 1

a_vec = np.random.randn(3,1)
b_vec = np.random.randn(1,5)

print(my_func(a_vec,b_vec))