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

# fly_num = 1544
fly_num = 1548

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()

#Look at T seconds of muscle activity

T = 100
fig = 1
dpt.plot_muscles_by_time(fig,flydf,T,fly.muscles_right,start_time=40,show_x_position=True)

trial = 'ol_blocks, g_x=-12, g_y=0, b_x=0, b_y=0, ch=1'
pre_trial_time = 3.
duration = 6.

fig = 2
flydf = dpt.baseline_by_prestim(flydf,fly.muscles)
dpt.plot_muscles_by_time(fig,flydf,T,fly.muscles_right,start_time=40,show_x_position=True)
plt.show()

dpt.plot_muscles_by_trial(flydf,trial,fly.muscles_phasic,pre_trial_time,
duration,show_kinematics=True,show_x_position=True)
plt.show()
