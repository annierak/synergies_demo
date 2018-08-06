from matplotlib import pyplot as plt
import numpy as np
import scipy
import flylib as flb
from flylib import util
import pandas as pd
import itertools
import time
import matplotlib.cm


fly_num = 1545
groups_of_interest = ['i','b']
side_of_interest = 'left' #Specify 'both' for both

frames_per_step = 5
disp_window_size = 5
corr_window_size = 1
duration = 10*60
start_at_stimulus = 'cl_blocks, g_x=-1, g_y=4, b_x=8, b_y=0, ch=True'

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

 #filter for L/R
if side_of_interest != 'both':
 filtered_muscle_cols = \
 [muscle for muscle in filtered_muscle_cols if util.muscle_side(muscle)==side_of_interest]

#filter for muscle type
filtered_muscle_cols = [
[muscle for muscle in filtered_muscle_cols if (
(group in muscle[0:len(group)]) and (muscle[len(group)+1]!='i'))]
for group in groups_of_interest]


group_lengths_list = [len(group_list) for group_list in filtered_muscle_cols]
num_groups = len(groups_of_interest)

filtered_muscle_cols = list(itertools.chain(*filtered_muscle_cols))
muscle_count = len(filtered_muscle_cols)

dt = frames_per_step*(flydf['t'][1]-flydf['t'][0])
corr_window_size = np.floor(corr_window_size/dt)*dt
disp_window_size = np.floor(disp_window_size/dt)*dt


stimulus_inds = flydf['stimulus']==start_at_stimulus

print(type(flydf['t'][stimulus_inds]))
print(flydf['t'][stimulus_inds][0:5])
print(flydf['t'][stimulus_inds][0:5].keys())

t = float(flydf['t'][stimulus_inds][0:1].tolist()[0])
counter = flydf['t'][stimulus_inds][0:1].keys()[0]
print(t,counter)

time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
muscle_matrix = np.array(flydf.loc[time_window_inds,filtered_muscle_cols]).T
steps_per_disp_window = int(disp_window_size/dt)
steps_per_corr_window = int(corr_window_size/dt)

muscle_pair_array = util.prod_list(filtered_muscle_cols,square=True)

muscle_pair_list = util.symm_matrix_half(muscle_pair_array)
group_index_list = [length*[index] for index,length in enumerate(group_lengths_list) ]

colormap1= matplotlib.cm.get_cmap('RdYlBu_r')
if len(groups_of_interest)>1:
    single_mucle_mode = True
    colormap2 = ['blue', 'orange', 'green', 'red',
     'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
else:
    single_mucle_mode = False
    pair_count = muscle_count*(muscle_count+1)/2 - muscle_count
    cmap = matplotlib.cm.get_cmap('Paired',pair_count)
    colormap2 = [cmap(ind) for ind in np.linspace(0,1,pair_count)]
num_colors = (num_groups*(num_groups+1)/2)

group_pair_list = util.symm_matrix_half(util.prod_list(groups_of_interest,square=True),exclude_diagonal=False)
group_pair_list = [tuple(pair) for pair in group_pair_list]
# print(group_pair_list)
# raw_input(' ')

if single_mucle_mode:
    color_list = np.zeros(np.shape(muscle_pair_list)[0]).astype(int)
    for ind,pair in enumerate(muscle_pair_list):
        chopped_pair = tuple(util.muscle_group(pair))
        try:
            color = group_pair_list.index(chopped_pair)
        except(ValueError):
            color = group_pair_list.index((chopped_pair[1],chopped_pair[0]))
        color_list[ind] = int(color)
else:
    color_list = range(pair_count)


#Test the muscle pair colors
# for i,color in enumerate(color_list):
#     if color==3:
#         print(muscle_pair_list[i])


plt.ion()
plt.figure(figsize=(11,8))
ax1 = plt.subplot(1,2,1)
title_text =ax1.set_title(' ',fontsize=2)

corr_values_by_t = np.zeros((muscle_count*(muscle_count+1)/2-muscle_count,steps_per_disp_window))
lines = plt.plot(corr_values_by_t.T)
for ind,line in enumerate(lines):
    line.set_color(colormap2[color_list[ind]])
corr_range = [-60,60]
plt.ylim(corr_range)
colors_used = [colormap2[unique_color] for unique_color in np.unique(color_list)]
custom_lines = [matplotlib.lines.Line2D([0], [0], color=c, lw=4) for c in colors_used]
if single_mucle_mode:
    legend_labs = group_pair_list
else:
    legend_labs = muscle_pair_list
plt.legend(custom_lines, legend_labs,loc=4,bbox_to_anchor=(1.6,-0.1  ,0.6,0.5),
    fontsize=10,ncol=2)
ax2 = plt.subplot(1,2,2)
image = plt.imshow(np.zeros((len(
    filtered_muscle_cols),len(filtered_muscle_cols))),vmin=corr_range[0],vmax=corr_range[1],interpolation='none',
cmap=colormap1)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('top')
plt.xticks(range(len(filtered_muscle_cols)),rotation=90)
plt.yticks(range(len(filtered_muscle_cols)))
ax2.set_xticklabels(filtered_muscle_cols)
ax2.set_yticklabels(filtered_muscle_cols)
ax2.set_position([0.55,0.40,0.45,0.45])

while t<duration:
    # print(counter)
    # print(t)
    text = flydf.iloc[counter]['stimulus']
    ax1.set_title(text,fontsize=12)
    # time.sleep(5)

    time_window_inds = (flydf['t']>t)&(flydf['t']<=t+corr_window_size)
    state_mtrx = np.array(flydf.loc[time_window_inds,filtered_muscle_cols]).T
    #Watch out for muscles that have no activity
    off_muscle_inds = (np.sum(state_mtrx,axis=1)==0.)
    # Set them to a very small amt of activity so nan's are not created
    state_mtrx[off_muscle_inds] = 1e-8
    centered_mtrx = state_mtrx - np.mean(state_mtrx,axis = 1)[:,None]
    std_mtrx = centered_mtrx/np.std(centered_mtrx,axis = 1)[:,None]
    cor_mtrx = np.dot(std_mtrx,std_mtrx.T)
    image.set_data(cor_mtrx)
    corr_values = util.symm_matrix_half(cor_mtrx)

    if counter<steps_per_disp_window:
        corr_values_by_t[:,counter] = corr_values
    else:
        corr_values_by_t[:,:-1] = corr_values_by_t[:,1:]
        corr_values_by_t[:,-1] = corr_values
        x_values = np.linspace(t,t+disp_window_size,steps_per_disp_window)
        ax1.set_xlim((t,t+disp_window_size))
        for row,line in enumerate(lines):
            line.set_ydata(corr_values_by_t[row,:])
            line.set_xdata(x_values)
            line.set_color(colormap2[color_list[row]])
    plt.draw()
    plt.pause(0.01)
    t+=dt
    counter+=1
