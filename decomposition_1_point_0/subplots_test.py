import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import gridspec

A = np.random.randn(4,100)

fig = plt.figure()
gs = gridspec.GridSpec(len(A),5)

for row in range(len(A)):
	print(row)
	plt.subplot(gs[row,:-1])
	plt.plot(A[row,:])
	plt.subplot(gs[row,-1])
	plt.plot(A[row,:])

plt.show()