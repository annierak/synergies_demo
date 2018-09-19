import substeps
import util
import synergy_estimators
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys

# ==========
# This script implements and tests the multiplicative update NMF algorithm for time-varying muscle synergies
# ==========

#constants
D = 5 #number of muscles
synergy_time = 1.  #Synergy duration
S = 50 #Number of episodes
T = 15    #synergy duration in time steps
N = 3 #number of synergies
display_episode = 0 #Which episode to display in plots/video

#Construct an M that is made of shifted Gaussians added together
# construct the synergies as Gaussians
variances = np.random.uniform(0.5*T/5,T/5,(N,D))
means = np.random.uniform(1,T,(N,D))
W = util.gauss_vector(T,means,variances)
#Scale the synergies with a different amplitude for each muscle/synergy time course
amp_max = 1
amplitudes = np.random.uniform(0,amp_max,(1,N,D))
c_min = 0
c_max = 1
W = amplitudes*W

plt.ion()
fig = plt.figure(1)

#Prepare video
if len(sys.argv)>1:
    video_name = sys.argv[1]
    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':video_name,}
    writer = FFMpegWriter(fps=10, metadata=metadata)
    writer.setup(fig, video_name+'.mp4', 500)
#Display the true W
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i,:].T,interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    pltuls.strip_ticks(ax)
plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)


#Construct true coeffients
true_c = np.random.uniform(c_min,c_max,(S,N))
plt.figure(1)
for i in range(N):
    ax = plt.gcf().get_axes()[i]
    ax.text(0.5,0.5,str(
        true_c[display_episode,i])[:5],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
W = np.moveaxis(W,0,2)

#Now make M by adding each of these scaled shifted N synergies muscle-wise
pre_sum_M = W[None,:,:,:]*true_c[:,:,None,None]
M = np.squeeze(np.sum(pre_sum_M,axis=1)) #M is S x D x T

#Construct estimator object
synergy_estimator = synergy_estimators.TimeDependentSynergyEstimator(
    S,D,T,N,M)
#Plot M and M_est
plt.figure(200)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.plot(M[display_episode,d,:])
plt.text(0.5,1,'Muscle Activity',transform=plt.gcf().transFigure)
lines = []
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.ylim([0,1])
    line, = plt.plot(synergy_estimator.M_est_stacked[display_episode,d,:])
    lines.append(line)
plt.figure(300)
im = plt.imshow(
    M[0,:,:],interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
pltuls.strip_ticks(ax)
plt.text(0.65,0.95,'True M',transform=plt.gcf().transFigure)

#Compute initial R^2
R2 = synergy_estimator.compute_R2()
print('R2 before starting: '+str(R2))

#Plot estimated W, c
ims = []
W_est_axes = []
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    W_est_axes.append(ax)
    im = plt.imshow(
        synergy_estimator.W_est_stacked[i,:,:],interpolation='none',
        aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    plt.colorbar()
    ims.append(im)
    pltuls.strip_ticks(ax)
plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)
c_texts = []
for i in range(N):
    max_W_est = np.max(synergy_estimator.W_est_stacked[i,:,:])
    max_W = np.max(W[i,:,:])
    c_est_scaled = synergy_estimator.c_est[display_episode,i]*max_W/max_W_est
    ax = W_est_axes[i]
    c_text = ax.text(0.5,0.5,str(
        c_est_scaled)[:3],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
    c_texts.append(c_text)


#Prepare shapes for updates
# stacked_M = np.copy(M)
# M = util.spread(M)  #reshape it so that it's D x T*S
# W_est = util.spread(W_est)


counter = 1
while abs(1.-synergy_estimator.compute_R2())>synergy_estimator.error_threshold:

    print('--------------ITERATION: '+str(counter)+'---------------')
    #Update delays and compute updated error
    synergy_estimator.update_delays() #size of delays (t) is S x N
    R2 = synergy_estimator.compute_R2()
    print('R2 after delay update: '+str(R2))

    #update H with new delays
    synergy_estimator.update_H()

    #update est. c and compute updated error
    synergy_estimator.update_c_est(regress=True)
    R2 = synergy_estimator.compute_R2()

    # print('c_est: ' + str(c_est[display_episode]))
    # print('true_c: ' + str(true_c[display_episode]))
    print('R2 after c update: '+str(R2))

    #Update H with new c's
    synergy_estimator.update_H()

    #update est W and compute updated error
    synergy_estimator.update_W_est()
    R2 = synergy_estimator.compute_R2()
    print('R2 after W update: '+str(R2))
    # print('delays: '+str(t))

    #Display current W_est estimates
    W_est = synergy_estimator.W_est_stacked

    #Match W_ests to true Ws
    if counter%5==1:
        true_syn_partners = substeps.match_synergy_estimates_td(W,W_est)
    #Display new W_ests
    for i in range(N):
    	im = ims[i]
        max_value = np.max(W_est[true_syn_partners[i],:,:])
    	im.set_data(util.normalize(W_est[true_syn_partners[i],:,:]))
        im.set_clim(vmin=0,vmax=amp_max)

    # Display the c_est, scaled to match the true_c magnitude
    plt.figure(1)
    for i in range(N):
        max_W_est = np.max(W_est[true_syn_partners[i],:,:])
        max_W = np.max(W[i,:,:])
        c_est_scaled = synergy_estimator.c_est[display_episode,true_syn_partners[i]]*max_W_est/max_W
        c_text = c_texts[true_syn_partners[i]]
        c_text.set_text(str(c_est_scaled)[:5])

    #video frame collection
    if len(sys.argv)>1:
        writer.grab_frame()

    #Display new M_est
    plt.figure(200)
    synergy_estimator.update_M_est() #********** this doesn't have t capacities yet
    M_est_ep = synergy_estimator.M_est_stacked[display_episode,:,:]
    for d in range(D):
        plt.subplot(D,1,d+1)
        line = lines[d]
        line.set_ydata(M_est_ep[d,:])
    plt.draw()
    plt.pause(0.02)
    counter+=1

plt.show()
raw_input(' ')
