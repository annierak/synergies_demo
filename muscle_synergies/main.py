import numpy as np
import matplotlib.pyplot as plt
import flylib as flb
import time
import itertools


fly= flb.NetFly(1380)

df = fly.construct_dataframe()

#print(df.columns.values)


muscles = np.array(['iii1_left', 'iii3_left',
 'i1_left',  'i2_left',
 'hg1_left', 'hg2_left', 'hg3_left', 'hg4_left',
 'b1_left', 'b2_left', 'b3_left',
 'iii1_right', 'iii3_right',
 'i1_right', 'i2_right',
 'hg1_right', 'hg2_right', 'hg3_right', 'hg4_right',
 'b1_right', 'b2_right', 'b3_right' ])

D = len(muscles)

synergy_time = 1  #Synergy duration

dt = (flydf['t'][1]-flydf['t'][0])  # time steps

T = np.ceil(D/dt)  #synergy duration in time steps

N = 5 #number of synergies

M = np.random.randn((D, T))  #randomnly generated muscle activity 

# M = util.generate_random_muscle_activity(D,T,N)  
# M = util.muscle_activity_empirical(T,muscles)

W = substeps.initialize_W(N,D,T)

c = substeps.initialize_c(N)

while error>error_threshold:
	delays = substeps.update_delay(M,W,c,T)
	c = substeps.update_c(c)
	W = substeps.update_W(W)
	error = substeps.compute_error(W,c,t,M)

