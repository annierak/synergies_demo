import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import figurefirst as fifi
import flylib as flb
import plotting_utls as pltutls

flynum = 1548

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()

filtered_muscle_cols = \
['iii1_left', 'iii3_left',
 'i1_left',  'i2_left',
 'hg1_left', 'hg2_left', 'hg3_left', 'hg4_left',
 'b1_left', 'b2_left', 'b3_left',
 'iii1_right', 'iii3_right',
 'i1_right', 'i2_right',
 'hg1_right', 'hg2_right', 'hg3_right', 'hg4_right',
 'b1_right', 'b2_right', 'b3_right' ]

trial_str = 'cl_blocks, g_x=-1, g_y=0, b_x=0, b_y=0, ch=1'
counter=np.where(flydf['stimulus']==trial_str)[0][0]
t=flydf.iloc[counter]['t']
