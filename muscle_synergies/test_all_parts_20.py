import substeps
import util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys




# ==========
# This piece is for testing the entire updating process (shifts, coefficients, Ws)
# ==========

D = 5 #number of muscles
synergy_time = 1.  #Synergy duration
S = 20
T = 15    #synergy duration in time steps
N = 3 #number of synergies

#We want to construct an M that is made of shifted Gaussians added together

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

#----This bit is for video making---------
if len(sys.argv)>1:
    video_name = sys.argv[1]
    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':video_name,}
    writer = FFMpegWriter(fps=10, metadata=metadata)
    writer.setup(fig, video_name+'.mp4', 500)
#------------------------------------------
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,0))
    plt.imshow(W[:,i,:].T,interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    pltuls.strip_ticks(ax)

plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

# plt.show()

#Make some coeffients c and multiply W by that coefficient for each synergy

display_episode = 0

true_c = np.random.uniform(c_min,c_max,(S,N))
plt.figure(1)
for i in range(N):
    ax = plt.gcf().get_axes()[i]
    ax.text(0.5,0.5,str(
        true_c[display_episode,i])[:5],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
W = np.moveaxis(W,0,2)

pre_sum_M = W[None,:,:,:]*true_c[:,:,None,None]

#Now make M by adding each of these scaled shifted N synergies muscle-wise
M = np.squeeze(np.sum(pre_sum_M,axis=1)) #M is S x D x T

W_est = substeps.initialize_W(N,D,T) #size of W is N x D x T
c_est = substeps.initialize_c(S,N) #size of c is S x N

#Plot M and M_est
plt.figure(200)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.plot(M[display_episode,d,:])
plt.text(0.5,1,'Muscle Activity',transform=plt.gcf().transFigure)
lines = []
M_est = np.sum(W_est[None,:,:,:]*c_est[:,:,None,None],axis=1)
for d in range(D):
    plt.subplot(D,1,d+1)
    plt.ylim([0,1])
    line, = plt.plot(M_est[display_episode,d,:])
    lines.append(line)

plt.figure(300)
im = plt.imshow(
    M[0,:,:],interpolation='none',
    aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
pltuls.strip_ticks(ax)
plt.text(0.65,0.95,'True M',transform=plt.gcf().transFigure)




Theta = np.zeros((N,2*T-1,N*T,T)) #shape of each Theta_i(t) is N*T x T


for i in range(1,N+1):
    for t in range(1-T,T):
        rows,columns = np.indices((N*T,T))
        to_fill = (rows+1-(i-1)*T)==(columns+1-t)
        to_fill[0:(i-1)*T,:] = 0.
        to_fill[i*T:,:] = 0.
        Theta[i-1,util.t_shift_to_index(t,T),:,:] = to_fill
        # plt.figure(544)
        # plt.imshow(Theta[i-1,util.t_shift_to_index(t,T),:,:],interpolation='none')
        # raw_input(' ')


# util.test_Theta(Theta)

error = np.inf

error_threshold = 1e-6
error = substeps.compute_squared_error(W_est,c_est,np.zeros_like(c_est),M)
print('error before starting: '+str(error))


ims = []
W_est_axes = []
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,2),(i,1))
    W_est_axes.append(ax)
    im = plt.imshow(
        W_est[i,:,:],interpolation='none',
        aspect=T/D,cmap='Greys_r',vmin=0,vmax=amp_max)
    plt.colorbar()
    ims.append(im)
    pltuls.strip_ticks(ax)

#Display scaled estimated c
c_texts = []
for i in range(N):
    max_W_est = np.max(W_est[i,:,:])
    max_W = np.max(W[i,:,:])
    c_est_scaled = true_c[display_episode,i]*max_W/max_W_est
    ax = W_est_axes[i]
    c_text = ax.text(0.5,0.5,str(
        c_est_scaled)[:3],color='purple',
        horizontalalignment='center',transform=ax.transAxes,
        fontsize=25)
    c_texts.append(c_text)


# plt.show()

plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

stacked_M = np.copy(M)
M = util.spread(M)  #reshape it so that it's D x T*S
W_est = util.spread(W_est)


counter = 1
while error>error_threshold:
    last = time.time()
    #Delay update
    delays = substeps.update_delay(stacked_M,util.stack(W_est,T),c_est,S) #size of delays (t) is S x N

    Theta_i_tis = Theta[0,util.t_shift_to_index(delays[0,0],T),:,:]

    error = substeps.compute_squared_error(util.stack(W_est,T),c_est,t,stacked_M)
    print('error after delay update: '+str(error))

    #update H with current delays
    H = util.construct_H(c_est,Theta,delays)

    #c update
    c_est = substeps.multiplicative_update_c(c_est,M,W_est,Theta,H,delays,scale=1)
    error = substeps.compute_squared_error(util.stack(W_est,T),c_est,delays,stacked_M)


    print('c_est: ' + str(c_est[display_episode]))
    print('true_c: ' + str(true_c[display_episode]))
    print('error after c update: '+str(error))

    #Update H with new c's
    H = util.construct_H(c_est,Theta,delays)
    #shape of H is N*T x S*T


    #W update
    W_est = substeps.multiplicative_update_W(M,W_est,H,scale=1)
    error = substeps.compute_squared_error(util.stack(W_est,T),c_est,delays,stacked_M)
    print('error after W update: '+str(error))
    print('delays: '+str(t))

    #Display current W_est estimates
    W_est = util.stack(W_est,T)

    if counter%5==1:
        #First, figure out what order the W_ests match up to the true Ws
        true_synergies = range(N)
        est_synergies = range(N)
        true_syn_partners = np.zeros(N)
        matching_scores = np.array(
        [[np.sum(np.abs(
            util.normalize(W[true_synergy,:,:])-
                util.normalize(W_est[est_synergy,:,:])))
             for est_synergy in est_synergies]
             for true_synergy in true_synergies])
        true_partners,est_partners = np.unravel_index(
            np.argsort(matching_scores,axis=None),np.shape(matching_scores))


        partners = np.array([true_partners,est_partners])
        print(partners)

        for i in range(N):
            true_syn_partners[partners[0,0]] = partners[1,0]
            cols_to_keep = (partners[1,:]!=partners[1,0]) & (partners[0,:]!=partners[0,0])
            partners = partners[:,cols_to_keep]

        true_syn_partners = true_syn_partners.astype('int')
        # raw_input(' ')

    print('TRUE SYN PARTNERS: '+str(true_syn_partners))
    #Then, display them
    for i in range(N):
    	im = ims[i]
        max_value = np.max(W_est[true_syn_partners[i],:,:])
    	im.set_data(util.normalize(W_est[true_syn_partners[i],:,:]))
        im.set_clim(vmin=0,vmax=amp_max)

    # Display the c_est, scaled to match the true_c magnitude
    plt.figure(1)
    for i in range(N):
        max_W_est = np.max(W_est[i,:,:])
        max_W = np.max(W[i,:,:])
        c_est_scaled = c_est[display_episode,i]*max_W_est/max_W
        c_text = c_texts[true_syn_partners[i]]
        c_text.set_text(str(c_est_scaled)[:5])

    if len(sys.argv)>1:
        writer.grab_frame()


    plt.figure(200)

    W_est = util.spread(W_est)

    #Display new M_est for display episode
    M_est_ep = W_est.dot(util.stack(H,T)[display_episode,:,:])
    for d in range(D):
        plt.subplot(D,1,d+1)
        line = lines[d]
        line.set_ydata(M_est_ep[d,:])
    plt.draw()
    plt.pause(0.02)
    print('--------------ITERATION: '+str(counter)+'---------------')
    counter+=1




plt.show()
