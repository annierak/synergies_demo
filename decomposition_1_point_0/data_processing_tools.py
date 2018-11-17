import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

def continuous_index_sets(indices):
    #Reshapes an array of indices with jumps
    #into a list of arrays such that each has no jumps
    split_inds = np.squeeze(np.where(np.diff(indices)>1.))
    split_inds = np.concatenate([np.array([-1]),split_inds,np.array([len(indices)])])
    inds_list = []
    for i in range(len(split_inds)-1):
        inds_list.append(indices[split_inds[i]+1:split_inds[i+1]])
    return inds_list

def ca_baseline_flight(muscle_array,filtered_muscle_cols,kinematics,time,plot=False):
    #For a time x muscle array, returns an estimate of the calcium
    #baseline value for each muscle

    #Uses a histogram of the calcium values, which is bimodal if the
    #fly stops flying during the trial: it has a smaller cluster of low
    #values corresponding to non-flight. The cutoff is the lower bound
    #of the larger upper cluster of the histogram.
    max_values = np.max(muscle_array,axis=0)
    min_values = np.min(muscle_array,axis=0)


    num_muscles = len(filtered_muscle_cols)

    if plot:
        plt.figure(2,figsize=(10,40))
        gs= matplotlib.gridspec.GridSpec(num_muscles+1,6)
    cutoffs = np.zeros((2,num_muscles))
    for i in range(num_muscles):
        if plot:
            plt.subplot(gs[i+1,-2])
            n,bins,_ = plt.hist(muscle_array[:,i],bins=30)
        else:
           n,bins = np.histogram(muscle_array[:,i],bins=30)
        diffs = np.diff(n)
        cutoff_index = np.argmax(diffs)-1
        #hg3 signal is behaving differently so account for that
        if filtered_muscle_cols[i][0:3]=='hg3':
            cutoffs[0,i]= 0
        else:
            cutoffs[0,i] = bins[cutoff_index]

        cutoffs[1,i] = np.percentile(muscle_array[:,i],99)


        if plot:
            ax=plt.subplot(gs[i+1,-1])
            plt.plot(n)
            plt.plot(np.diff(n))
            plt.subplot(gs[i+1,:-2])
            plt.plot(time,muscle_array[:,i])
            plt.plot(time,cutoffs[0,i]*np.ones_like(muscle_array[:,i]),color='r')
            plt.plot(time,cutoffs[1,i]*np.ones_like(muscle_array[:,i]),color='r')
            plt.ylabel(filtered_muscle_cols[i])
            plt.ylim([min_values[i],max_values[i]])
            plt.tight_layout()
            ax.set_yticks([])
            ax.set_yticklabels('')

    if plot:
        plt.subplot(gs[0,:-2])
        plt.plot(kinematics)

    return cutoffs

def raw_ca_to_dff(flydf):
    #Revised version for processing raw calcium signal.
    kinematics = flydf['amp_diff'].values
    non_flight_frames = np.where(np.isnan(kinematics))
    # print(np.shape(non_flight_frames))
    # plt.plot(np.squeeze(non_flight_frames),'o')
    # plt.show()
    # baseline_F = flydf.[]

def baseline_by_prestim(flydf,muscles):
    inds = flydf['pre_stimulus_rest_bool'] == 1
    prestim_means_by_muscle = flydf.loc[inds, muscles].mean()
    print(prestim_means_by_muscle)
    flydf[muscles] = flydf[muscles] - prestim_means_by_muscle
    for muscle in muscles:
        flydf.loc[flydf[muscle]<0.,muscle] = 0.
    return flydf

def plot_muscles_by_time(fig,flydf,end_time,which_muscles,start_time=0.,
    show_kinematics=True,show_x_position=False,show_y_position=False):
    end_index = np.where(flydf['t']>=end_time)[0][0]
    start_index = np.where(flydf['t']>=start_time)[0][0]
    time_window_inds = np.arange(start_index,end_index)
    times = flydf['t'][time_window_inds]

    muscle_array = flydf.loc[time_window_inds,which_muscles].values

    max_values = np.max(muscle_array,axis=0)
    min_values = np.min(muscle_array,axis=0)

    num_muscles = len(which_muscles)

    plt.figure(fig,figsize=(10,40))
    extra_rows = show_kinematics+show_x_position+show_y_position
    gs= matplotlib.gridspec.GridSpec(num_muscles+extra_rows,6)
    for i in range(num_muscles):
        ax = plt.subplot(gs[i,:])
        plt.plot(times,muscle_array[:,i])
        plt.ylabel(which_muscles[i],rotation=0,position=(0.,0.),#transform=plt.gca().transAxes,
            color='r')
        plt.ylim([min_values[i],max_values[i]])
        # plt.tight_layout()
        plt.subplots_adjust(left=0.1,right=0.9,top=.95,bottom=.05,hspace=0.3)
        # ax.set_yticks([])
        # ax.set_yticklabels('')
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.05, 0.5)

    plot_row = num_muscles

    if show_kinematics:
        kinematics = flydf.loc[time_window_inds,['amp_diff']].values
        ax = plt.subplot(gs[plot_row,:])
        ax.set_yticks([])
        ax.set_yticklabels('')
        plt.plot(times,kinematics)
        plt.ylabel('Kinematics',rotation=0)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.05, 0.5)
        plot_row+=1

    if show_x_position:
        x_pos = flydf.loc[time_window_inds,['x_pos']].values
        ax = plt.subplot(gs[plot_row,:])
        ax.set_yticks([])
        ax.set_yticklabels('')
        plt.plot(times,x_pos)
        plt.ylabel('x pos',rotation=0)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.05, 0.5)
        plot_row+=1

    if show_y_position:
        kinematics = flydf.loc[time_window_inds,['y_pos']].values
        ax = plt.subplot(gs[plot_row,:])
        ax.set_yticks([])
        ax.set_yticklabels('')
        plt.plot(times,y_pos)
        plt.ylabel('y pos',rotation=0)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_label_coords(1.05, 0.5)
        plot_row+=1



    # plt.show()



def plot_muscles_by_trial(flydf,trial,which_muscles,pre_trial_time,duration,
    show_kinematics=True,show_x_position=False,show_y_position=False):

    trial_inds=np.squeeze(np.where(flydf['stimulus']==trial))
    trial_inds_list = continuous_index_sets(trial_inds)
    num_episodes = len(trial_inds_list)
    num_muscles = len(which_muscles)
    dt = (flydf['t'][1]-flydf['t'][0])
    extra_rows = show_kinematics+show_x_position+show_y_position
    gs= matplotlib.gridspec.GridSpec(num_muscles+extra_rows,6)


    for j,trial_inds in enumerate(trial_inds_list):
        plt.figure(100+j,figsize=(5,20))

        trial_start_index = trial_inds[0]
        time_window_inds = np.arange(trial_start_index + int(np.floor(
            ((3-pre_trial_time)/dt))), trial_start_index + int(np.floor(
                (3+duration)/dt)))
        times =  flydf['t'][time_window_inds].values


        kinematics = flydf.loc[time_window_inds,['amp_diff']].values.squeeze()
        muscle_array = flydf.loc[time_window_inds,which_muscles].values

        max_values = np.max(muscle_array,axis=0)
        min_values = np.min(muscle_array,axis=0)

        for i in range(num_muscles):
            ax = plt.subplot(gs[i,:])
            plt.plot(times,muscle_array[:,i])
            plt.ylabel(which_muscles[i],rotation=0,position=(0.,0.),#transform=plt.gca().transAxes,
                color='r')
            plt.ylim([min_values[i],max_values[i]])
            # plt.tight_layout()
            plt.subplots_adjust(left=0.01,right=0.8,top=.95,bottom=.05,hspace=0.3)
            ax.set_yticks([])
            ax.set_yticklabels('')
            ax.set_xlim([times[0],times[-1]])
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.12, 0.5)

        plot_row = num_muscles

        if show_kinematics:
            ax = plt.subplot(gs[plot_row,:])
            ax.set_yticks([])
            ax.set_yticklabels('')
            plt.plot(times,kinematics)
            plt.ylabel('Kinematics',rotation=0)
            ax.set_xlim([times[0],times[-1]])
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.12, 0.5)
            plot_row+=1

        if show_x_position:
            x_pos = flydf.loc[time_window_inds,['x_pos']].values
            ax = plt.subplot(gs[plot_row,:])
            ax.set_yticks([])
            ax.set_yticklabels('')
            plt.plot(times,x_pos)
            ax.set_xlim([times[0],times[-1]])
            plt.ylabel('x pos',rotation=0)
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.12, 0.5)
            plot_row+=1

        if show_y_position:
            kinematics = flydf.loc[time_window_inds,['y_pos']].values
            ax = plt.subplot(gs[plot_row,:])
            ax.set_yticks([])
            ax.set_yticklabels('')
            plt.plot(times,y_pos)
            plt.ylabel('y pos',rotation=0)
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.05, 0.5)
            plot_row+=1


    # plt.show()
