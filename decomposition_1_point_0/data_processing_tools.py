import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def ca_baseline_flight(muscle_array,filtered_muscle_cols,plot=False):
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
        gs= matplotlib.gridspec.GridSpec(num_muscles,6)
    cutoffs = np.zeros(2,num_muscles)
    for i in range(num_muscles):
        if plot:
            plt.subplot(gs[i,-2])
            n,bins,_ = plt.hist(muscle_array[:,i],bins=30)
        else:
           n,bins = np.histogram(muscle_array[:,i],bins=30)
        diffs = np.diff(n)
    	cutoff_index = np.argmax(diffs)
        #hg3 signal is behaving differently so account for that
    	if filtered_muscle_cols[i][0:3]=='hg3':
    		cutoffs[0,i]= 0
    	else:
    		cutoffs[0,i] = bins[cutoff_index]

        cutoffs[1,i] = np.percentile(muscle_array[:,i],95)


        if plot:
        	ax=plt.subplot(gs[i,-1])
        	plt.plot(n)
        	plt.plot(np.diff(n))
        	plt.subplot(gs[i,:-2])
        	plt.plot(muscle_array[:,i])
            plt.plot(cutoffs[0,i]*np.ones_like(muscle_array[:,i]),color='r')
            plt.plot(cutoffs[1,i]*np.ones_like(muscle_array[:,i]),color='r')
            plt.ylabel(filtered_muscle_cols[i])
        	plt.ylim([min_values[i],max_values[i]])
        	plt.tight_layout()
        	ax.set_yticks([])
        	ax.set_yticklabels('')


    return cutoffs
