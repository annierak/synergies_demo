#October 4 2018
#Efforts to speedup plotting and access of fly muscle data
#Adding/testing features of the flylib Fly object

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import flylib as flb
from matplotlib import gridspec
import data_processing_tools as dpt
import sys

fly_num = 1544
# fly_num = 1548

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()

#Look at first T seconds of muscle activity
T = 600
dpt.plot_muscles_by_time(flydf,T,fly.muscles_phasic)
