from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.signal

import flylib as flb

from flylib import util
import local_project_functions as lpf

# import figurefirst as fifi
# from IPython.display import SVG,display
# import networkx as nx

fly = flb.NetFly(1548,rootpath='/home/annie/work/programming/fly_muscle_data/')

print(type(fly))
print(fly.__dict__.keys())

fly.open_signals()
print(fly.__dict__.keys())

right_calc = fly.ca_cam_right

print(type(right_calc))
print(type(fly.time))
print(fly.time[1:10])
a = np.array(fly.time[1:10])

name,signal = fly.ca_cam_left_model_fits.items()[3]
print(type(signal))

plt.plot(signal[0:1000])
# plt.show()


