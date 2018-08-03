import matplotlib.pyplot as plt
import numpy as np

a = np.random.randn(5,100)

counter = 0

plt.ion()
plt.figure()
# plt.show()
lines = plt.plot(a.T)
print([line for line in lines])

for i in range(100):
    a[:,:-1] = a[:,1:]
    a[:,-1] = np.random.randn(5)
    for row,line in enumerate(lines):
        print(a[row,:])
        line.set_ydata(a[row,:])
        plt.draw()
    plt.pause(.1)
