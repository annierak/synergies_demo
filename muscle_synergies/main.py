import numpy as np
import matplotlib.pyplot as plt
import flylib as flb
import time
import itertools
import substeps

fly_num = 1380

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)
flydf = fly.construct_dataframe()

#print(df.columns.values)


muscles = np.array(['iii1_left', 'iii3_left',
 'i1_left',  'i2_left',
 'hg1_left', 'hg2_left', 'hg3_left', 'hg4_left',
 'b1_left', 'b2_left', 'b3_left',
 'iii1_right', 'iii3_right',
 'i1_right', 'i2_right',
 'hg1_right', 'hg2_right', 'hg3_right', 'hg4_right',
 'b1_right', 'b2_right', 'b3_right' ])

S = 4 #number of episodes, indexed by s

D = len(muscles) #number of muscles

synergy_time = 1.  #Synergy duration

dt = (flydf['t'][1]-flydf['t'][0])  # time steps

T = int(np.ceil(synergy_time/dt))  #synergy duration in time steps

N = 5 #number of synergies

M = np.random.randn(S, D, T)  #randomly generated muscle activity

# M = util.generate_random_muscle_activity(D,T,N)
# M = util.muscle_activity_empirical(T,muscles)

W = substeps.initialize_W(N,D,T) #size of W is N x D x T

c = substeps.initialize_c(S,N) #size of c is S x N

error = np.inf

error_threshold = 1e-6

while error>error_threshold:
	last = time.time()
	delays = substeps.update_delay(M,W,c,S) #size of delays (t) is S x N
	print(time.time()-last)
	raw_input('here')
	c = substeps.update_c(c)
	W = substeps.update_W(W)
	error = substeps.compute_error(W,c,t,M)
