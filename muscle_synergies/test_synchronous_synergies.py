import substeps
import util
import numpy as np
import matplotlib
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys


D = 5 #number of muscles
S = 50 #number of episodes
T = 120    #time samples
N = 3 #number of synergies

#Generate coeffients

C = np.random.uniform(0,1,(N,T))
W = np.random.uniform(0,1,(D,N))

M = np.dot(W,C)

plt.ion()
plt.figure(1)

colormap = matplotlib.cm.get_cmap('Reds')
for i in range(N):
    ax = plt.subplot2grid((N,3),(i,0))
    plt.imshow(util.normalize(W[:,i].T[:,None]),interpolation='none',
    aspect=3./D,cmap=colormap,vmin=0,vmax=1)
    pltuls.strip_bare(ax,axis='x')
    pltuls.strip_ticks(ax,axis='y')


plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)


# plt.show()

C_est = np.random.uniform(0,1,(N,T))
W_est = np.random.uniform(0,1,(D,N))


#Display the W_ests
ims = []
# W_est_axes = []
plt.figure(1)
for i in range(N):
    ax = plt.subplot2grid((N,3),(i,2))
    # W_est_axes.append(ax)
    im = plt.imshow(W_est[:,i].T[:,None],interpolation='none',
    aspect=3./D,cmap=colormap,vmin=0,vmax=1)
    pltuls.strip_bare(ax,axis='x')
    pltuls.strip_ticks(ax,axis='y')
    ims.append(im)

bar_width = 0.35
barcollections = []
for i in range(N):
	ax = plt.subplot2grid((N,3),(i,1))
	plt.bar(np.arange(D),util.normalize(W[:,i]),bar_width,color='r')
	barcollection = plt.bar(np.arange(D)+bar_width,util.normalize(W_est[:,i]),bar_width,color='b')
	plt.ylim([0,1.5])
	pltuls.strip_bare(ax,axis='y')
	barcollections.append(barcollection)


plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

plt.figure(2)
plt.subplot(2,1,1)
plt.imshow(M,interpolation='none',cmap=colormap,vmin=0,vmax=1)
plt.title('True M')

M_est = np.dot(W_est,C_est)
plt.subplot(2,1,2)
im1 = plt.imshow(M_est,interpolation='none',cmap=colormap,vmin=0,vmax=1)
plt.title('Estim. M')



error = substeps.compute_error_by_trace(M,W_est,C_est)

print('error: '+str(error))

error_threshold = 1e-3

plt.show()

counter = 1
while error > error_threshold:
	print('----ITERATION----'+str(counter))
	print('error: '+str(error))
	C_est = substeps.mult_update_c_synchronous(M,W_est,C_est)
	W_est = substeps.mult_update_W_synchronous(M,W_est,C_est)

	if counter%5==1:
    #First, figure out what order the W_ests match up to the true Ws
   		true_synergies = range(N)
    	est_synergies = range(N)
    	true_syn_partners = np.zeros(N)
    	matching_scores = np.array(
    	[[np.sum(np.abs(
	        util.normalize(W[:,true_synergy])-
	            util.normalize(W_est[:,est_synergy])))
	         for est_synergy in est_synergies]
	         for true_synergy in true_synergies])
    	true_partners,est_partners = np.unravel_index(
	        np.argsort(matching_scores,axis=None),np.shape(matching_scores))
    	partners = np.array([true_partners,est_partners])
		# print(partners)
	for i in range(N):
		true_syn_partners[partners[0,0]] = partners[1,0]
		cols_to_keep = (partners[1,:]!=partners[1,0]) & (partners[0,:]!=partners[0,0])
		partners = partners[:,cols_to_keep]
		true_syn_partners = true_syn_partners.astype('int')

	for i in range(N):
		im = ims[i]
		im.set_data(util.normalize(W_est[:,true_syn_partners[i]]).T[:,None])

	for i in range(N):
		barcollection = barcollections[i]
		W_est_i_normed = util.normalize(W_est[:,true_syn_partners[i]])
		for d,bar in enumerate(barcollection):
			bar.set_height(W_est_i_normed[d])

	M_est = np.dot(W_est,C_est)
	im1.set_data(M_est)
	plt.pause(0.1)
	error = substeps.compute_error_by_trace(M,W_est,C_est)
	counter+=1

raw_input(' ')