import numpy as np
from flylib import util

a = np.random.randn(5,5)

print(a)
print(util.symm_matrix_half(a))
