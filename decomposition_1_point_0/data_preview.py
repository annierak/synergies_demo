import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import figurefirst as fifi
import flylib as flb
from matplotlib import gridspec
import data_processing_tools as dpt

fly_num = 1548
# fly_num = 1549

try:
    fly = flb.NetFly(fly_num,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
except(OSError):
    fly = flb.NetFly(fly_num)

flydf = fly.construct_dataframe()

filtered_muscle_cols = \
['iii1_left',
 'i1_left',
 'hg1_left', 'hg2_left', 'hg3_left',
  'b2_left',
 'iii1_right',
 'i1_right',
 'hg1_right', 'hg2_right', 'hg3_right',
 'b2_right' ]

num_muscles = len(filtered_muscle_cols)

dt = (flydf['t'][1]-flydf['t'][0])


#Look at first T seconds of muscle activity

T = 600.
end_index = np.where(flydf['t']>=T)[0][0]
start_index = 0
time_window_inds = np.arange(start_index,end_index)
times = flydf['t'][time_window_inds]

kinematics = flydf.loc[time_window_inds,['amp_diff']].values
muscle_array = flydf.loc[time_window_inds,filtered_muscle_cols].values

max_values = np.max(muscle_array,axis=0)
min_values = np.min(muscle_array,axis=0)





#For each muscle, find the minimum calcium activity during flight

cutoffs = dpt.ca_baseline_flight(muscle_array,filtered_muscle_cols)

# print(np.min(muscle_array))
# plt.figure(4)
# plt.hist(muscle_array[:,1])
# plt.show()
# raw_input(' ')

muscle_array = muscle_array - np.array(cutoffs)[None,:]
muscle_array[muscle_array<0.] = 0.

# plt.figure(3)

# for i,muscle in enumerate(filtered_muscle_cols):
# 	ax = plt.subplot(gs[i+1,:-2])
# 	plt.plot(times,muscle_array[:,i])
# 	ax.invert_yaxis()
# 	plt.ylabel(muscle)
# 	# ax.yaxis.set_label_position("right")
# 	# plt.subplots_adjust(hspace=1.5)
# 	plt.tight_layout()
# 	ax.set_yticks([])
# 	ax.set_yticklabels('')





#This part displays the interval -2 to +7 of the specified trial
trial_str = 'ol_blocks, g_x=-12, g_y=0, b_x=0, b_y=0, ch=1'
onset_duration = 7.
pre_onset_time = 2.

start_index=np.where(flydf['stimulus']==trial_str)[0][0]
time_window_inds = np.arange(start_index - int(np.floor(pre_onset_time/dt)), start_index + int(np.floor(onset_duration/dt)))
times = flydf['t'][time_window_inds]

kinematics = flydf.loc[time_window_inds,['amp_diff']].values
muscle_array = flydf.loc[time_window_inds,filtered_muscle_cols].values

muscle_array = muscle_array - np.array(cutoffs)[None,:]
muscle_array[muscle_array<0.] = 0.

max_values = np.max(muscle_array,axis=0)
min_values = np.min(muscle_array,axis=0)



plt.figure(1,figsize=(10,40))

for i,muscle in enumerate(filtered_muscle_cols):
	ax = plt.subplot(num_muscles+1,1,i+2)
	plt.plot(times,muscle_array[:,i])
	ax.invert_yaxis()
	plt.ylabel(muscle)
	ax.yaxis.set_label_position("right")
	# plt.ylim([min_values[i],max_values[i]])
	# plt.subplots_adjust(hspace=1.5)
	plt.tight_layout()
	ax.set_yticks([])
	ax.set_yticklabels('')
ax = plt.subplot(num_muscles+1,1,1)
plt.ylabel('Kinematics')
plt.plot(times,kinematics)
ax.yaxis.set_label_position("right")
ax.set_yticks([])
ax.set_yticklabels('')


plt.show()
