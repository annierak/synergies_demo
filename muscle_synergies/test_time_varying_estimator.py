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

#reproducible results
np.random.seed(6)

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
max_W_value = np.max(W)
true_W_ims = []
true_W_axes = []
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    W_im = plt.imshow(W[:,i,:].T,interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=max_W_value)
    plt.colorbar()
    true_W_axes.append(ax)
    true_W_ims.append(W_im)
    pltuls.strip_ticks(ax)
plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)


#Construct true coeffients
true_c = np.random.uniform(c_min,c_max,(S,N))
#Construct true delays
true_delays = np.floor(np.random.uniform(-T/2,T/2,(S,N))).astype(int)
# true_delays = np.zeros((S,N)).astype(int)
print('delays: '+str(true_delays[display_episode,:]))

plt.figure(1)
for i in range(N):
    ax = true_W_axes[i]
    ax.text(0.5,0.5,str(
        true_c[display_episode,i])[:5],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
    ax.text(-0.3,0.5,str(
        true_delays[display_episode,i])[:5],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=15)

W = np.moveaxis(W,0,2)


# #Test the new shifting function
# plt.figure(667)
# plt.subplot(2,1,1)
# plt.imshow(W[0,:,:],interpolation='none')
# shifts = [-20,-7,6,20,50]
# for shift in shifts:
#     plt.subplot(2,1,2)
#     plt.clf
#     plt.imshow(substeps.shift_matrix_columns_2(shift,W[0,:,:],2*T),interpolation='none')
#     plt.title('Shift of '+str(shift))
#     plt.show()
#     raw_input(' ')

# shift W according to selected delays
shifted_W = np.array(
    [[substeps.shift_matrix_columns_2(int(true_delays[s,i]),W[i,:,:],2*T) \
        for i in range(N)] for s in range(S)])


#Now make M by adding each of these scaled shifted N synergies muscle-wise
pre_sum_M = shifted_W[:,:,:,:]*true_c[:,:,None,None]


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
#Display muscle Activity
# plt.figure(300)
# im = plt.imshow(
#     M[0,:,:],interpolation='none',
#     aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
# pltuls.strip_ticks(ax)
# plt.text(0.65,0.95,'True M',transform=plt.gcf().transFigure)

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
c_texts = [];delay_texts = []
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
    delay_text = ax.text(-0.3,0.5,str(
        synergy_estimator.delays[display_episode,i])[:3],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=15)
    delay_texts.append(delay_text)


plt.figure(333)
cax = plt.subplot(1,2,1)
c_dots = plt.scatter(true_c.flatten(),synergy_estimator.c_est.flatten())
plt.plot([0,1],[0,1],color='r',linewidth=3)
plt.title('C_est vs True C')
tax = plt.subplot(1,2,2)
t_dots = plt.scatter(true_delays.flatten(),synergy_estimator.delays.flatten())
plt.plot([-(2./3)*T,(2./3)*T],[-(2./3)*T,(2./3)*T],color='r',linewidth=3)
plt.ylim([-T,T])
plt.xlim([-T,T])
plt.title('Est Delay vs True Delay')
cax.set_aspect('equal')
tax.set_aspect('equal')
#Prepare shapes for updates
# stacked_M = np.copy(M)
# M = util.spread(M)  #reshape it so that it's D x T*S
# W_est = util.spread(W_est)


counter = 1
while abs(1.-synergy_estimator.compute_R2())>synergy_estimator.error_threshold:

    print('--------------ITERATION: '+str(counter)+'---------------')
    # debug = (counter>5)
    debug = False
    #Update delays and compute updated error
    synergy_estimator.update_delays(debug=debug) #size of delays (t) is S x N
    R2 = synergy_estimator.compute_R2()
    print('R2 after delay update: '+str(R2))

    #update H with new delays
    synergy_estimator.update_H()

    #update est. c and compute updated error
    synergy_estimator.update_c_est()
    R2 = synergy_estimator.compute_R2()

    # print('c_est: ' + str(c_est[display_episode]))
    # print('true_c: ' + str(true_c[display_episode]))
    print('R2 after c update: '+str(R2))

    #Update H with new c's
    synergy_estimator.update_H()

    #update est W and compute updated error
    synergy_estimator.update_W_est(normalize=False)
    R2 = synergy_estimator.compute_R2()
    print('R2 after W update: '+str(R2))
    # print('delays: '+str(t))

    #Display current W_est estimates
    W_est = synergy_estimator.W_est_stacked

    #Match W_ests to true Ws
    if counter%5==1:
        true_syn_partners = substeps.match_synergy_estimates_td(W,W_est)
    #Display new W_ests
    max_value = np.max(W_est)
    # for im in true_W_ims:
    #     im.set_clim(vmin=0,vmax=max_value )
    for i in range(N):
    	im = ims[i]
    	# im.set_data(util.normalize(W_est[true_syn_partners[i],:,:]))
    	im.set_data(W_est[true_syn_partners[i],:,:])
        im.set_clim(vmin=0,vmax=max_value)

    # Display the c_est, scaled to match the true_c magnitude

    max_W_est = np.max(W_est[true_syn_partners,:,:],axis=(1,2))
    max_W = np.max(W,axis=(1,2))
    c_est_scaled = synergy_estimator.c_est[:,true_syn_partners]*max_W_est[None,:]/max_W[None,:]
    #This version is the scaling the c_est to the value it would be if its W_est had a peak matching the peak of the true W
    plt.figure(1)
    for i in range(N):
        c_text = c_texts[i]
        c_text.set_text(str(c_est_scaled[display_episode,i])[:5])

        #Debugging------
        # max_W_est = np.max(synergy_estimator.W_est_stacked[true_syn_partners[i],:,:])
        # max_W = np.max(W[i,:,:])
        # c_est_scaled = synergy_estimator.c_est[display_episode,true_syn_partners[i]]*max_W_est/max_W
        # c_text.set_text(str(c_est_scaled)[:5])
        #----------

        delay_text = delay_texts[i]
        delay = int(synergy_estimator.delays[display_episode,true_syn_partners[i]])
        delay_text.set_text(str(delay)[:5])

    #c and delays scatter plot
    c_dots.set_offsets(np.c_[true_c.flatten(),c_est_scaled.flatten()])
    t_dots.set_offsets(np.c_[true_delays.flatten(),synergy_estimator.delays[:,true_syn_partners].flatten()])


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
