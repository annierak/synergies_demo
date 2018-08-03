import numpy as np
import matplotlib.pyplot as plt

image = np.random.randn(10,10)
xs = range(10)
ys = np.random.randn(10)

plt.ion()

plt.figure(1)
plt.subplot(2,1,1)
im = plt.imshow(image)
plt.subplot(2,1,2)
line, = plt.plot(xs,ys)
print(line)
for i in range(100):
    print i
    image = np.random.randn(10,10)
    im.set_data(image)
    line.set_ydata(np.random.randn(10))
    plt.draw()
    plt.pause(.1)
