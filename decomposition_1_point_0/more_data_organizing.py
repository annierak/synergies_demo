import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import figurefirst as fifi
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

if len(np.unique(flydf['t']))!=len(flydf['t']):
    print('Watch out time stamps are not unique')
    sys.exit()

T = np.max(flydf['t'])
end_index = np.where(flydf['t']>=T)[0][0]
start_index = 0
time_window_inds = np.arange(start_index,end_index+1)
times = flydf['t'][time_window_qinds]

kinematics = flydf.loc[time_window_inds,['amp_diff']].values
muscle_array = flydf.loc[time_window_inds,filtered_muscle_cols].values

# flydf = dpt.raw_ca_to_dff(flydf)
# raw_input(' ')

cutoffs = dpt.ca_baseline_flight(muscle_array,filtered_muscle_cols,kinematics,times,plot=True)
# plt.show()

muscle_array = muscle_array - np.array(cutoffs[0,:])[None,:]
muscle_array[muscle_array<0.] = 0.
muscle_maxes = np.max(muscle_array,axis=0)
print(np.shape(muscle_maxes))


#Begin filtering for specific trial type
trial_str = 'ol_blocks, g_x=0, g_y=4, b_x=0, b_y=0, ch=1'
static_time = 2.
motion_length = 50. #Time after stimulus onset selected to analyze


trial_inds=np.squeeze(np.where(flydf['stimulus']==trial_str))
split_inds = np.squeeze(np.where(np.diff(trial_inds)>1.))
split_inds = np.concatenate([np.array([-1]),split_inds,np.array([len(trial_inds)])])

trial_inds_list = []

for i in range(len(split_inds)-1):
    trial_inds_list.append(trial_inds[split_inds[i]+1:split_inds[i+1]])

for trial_inds_set in trial_inds_list:
    # print(len(trial_inds_set))
    print(np.unique(flydf['stimulus'][trial_inds_set]))

num_episodes = len(trial_inds_list)

trial_activity_array = np.zeros((num_episodes,num_muscles,int(np.ceil(motion_length/dt))))
trial_times = np.zeros((num_episodes,int(np.ceil(motion_length/dt))))
trial_kinematics = np.zeros_like(trial_times)


for j,trial_inds in enumerate(trial_inds_list):
    #Fill trial_activity_array with baseline subtracted Ca data for each trial
    trial_start_index = trial_inds[0]
    time_window_inds = np.arange(trial_start_index + int(np.floor(
        static_time/dt)), trial_start_index + int(np.floor(
            (static_time+motion_length)/dt)))
    times = flydf['t'][time_window_inds]
    trial_times[j,:] = times

    trial_kinematics[j,:] = flydf.loc[time_window_inds,['amp_diff']].values.squeeze()
    muscle_array = flydf.loc[time_window_inds,filtered_muscle_cols].values

    muscle_array = muscle_array - np.array(cutoffs[0,:])[None,:]
    muscle_array[muscle_array<0.] = 0.

    trial_activity_array[j,:,:] = muscle_array.T


#Normalize data to max of 1 per muscle per trial
trial_activity_array = trial_activity_array/(muscle_maxes[None,:,None])
trial_activity_array[np.isnan(trial_activity_array)] = 0.


for j,trial_inds in enumerate(trial_inds_list):

    plt.figure(100+j,figsize=(5,20))

    for i,muscle in enumerate(filtered_muscle_cols):
        ax = plt.subplot(num_muscles+1,1,i+2)
        plt.plot(trial_times[j,:],trial_activity_array[j,i,:])
        plt.ylim([0,1])
        plt.ylabel(muscle,rotation=0,horizontalalignment='right',position=(4,1),
            color='r')
        ax.yaxis.set_label_position("right")
        plt.tight_layout()
        # ax.set_yticks([])
        # ax.set_yticklabels('')
    ax = plt.subplot(num_muscles+1,1,1)
    plt.ylabel('Kinematics')
    plt.plot(trial_times[j,:],trial_kinematics[j,:])
    ax.yaxis.set_label_position("right")


#Take mean across time
avg_trial_activity_array = np.mean(trial_activity_array,axis=2)
print(np.shape(avg_trial_activity_array))
plt.figure(666)
fig, ax = plt.subplots()
plt.imshow(avg_trial_activity_array.T,interpolation='none')
ax.set_yticks(range(num_muscles))
ax.set_yticklabels(filtered_muscle_cols)
plt.show()
