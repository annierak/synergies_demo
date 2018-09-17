#This script replicates the synergy extraction effectiveness test presented in
#Tresch, Cheung, d'Avella 2006
#specifically just for the nonnegative matrix factorization algorithm
#that is described in d'Avella and Bizzi 2005 supplemental text

import substeps
import util
import numpy as np
import matplotlib
import matplotlib.animation as animate
import matplotlib.pyplot as plt
import plotting_utls as pltuls
import time
import sys
import scipy.linalg


#Constants used for the script
num_reps = 25
D = 12 #number of muscles
T = 1000    #time samples
N = 4 #number of synergies
lam = 10. #lambda for exponential distribution of true W, C
oneminusrsq = 5. #initial value of 1 - R^2 for repetition process
a  = 3e-4 # noise "slope" -- see Tresch et al 2006, pg 2200
error_threshold = 5e-4

#Initialize similarity comparision variables (Tresch pg 2202)
subspace_similarities_b_s = np.zeros((2,num_reps))
basis_vector_similarities_b_s = np.zeros((2,num_reps))
coefficient_similarities_b_s = np.zeros((2,num_reps))

for k in range(num_reps): #Each rep is a new iteration of the algorithm
    while (oneminusrsq>0.15) or (oneminusrsq<.12):
        #Create true W and C
        print('redrawing to reduce noise')
        W = np.random.exponential(1./lam,(D,N))
        C = np.random.exponential(1./lam,(N,T))
        #Add noise to W & C and sum to create M
        raw_sd = np.sum(np.dot(W,C),axis=1)
        raw_sd[raw_sd<0] = 0
        muscle_noise_vector = a*raw_sd #noise is D x 1
        M = np.dot(W,C) # M is D x N
        M_nsy = M+muscle_noise_vector[:,None]
        #Choose a s.t. the 1-R^2 for the original explaining the noise is ~15%
        SS_res = np.sum(np.square(M_nsy-M))
        SS_tot = np.sum(np.square(M_nsy-np.mean(M_nsy)))
        oneminusrsq = SS_res/SS_tot
    print("1 - R^2: "+str(oneminusrsq))

    plt.ion()
    #Plot the true W
    plt.figure(1)
    colormap = matplotlib.cm.get_cmap('Reds')
    for i in range(N):
        ax = plt.subplot2grid((N,3),(i,0))
        plt.imshow(util.normalize(W[:,i].T[:,None]),interpolation='none',
        aspect=3./D,cmap=colormap,vmin=0,vmax=1)
        pltuls.strip_bare(ax,axis='x')
        pltuls.strip_ticks(ax,axis='y')
    plt.text(0.25,0.95,'True W',transform=plt.gcf().transFigure)

    #Initialize estimated C and W (which designate M_est)
    C_est = np.random.uniform(0,1,(N,T))
    W_est = np.random.uniform(0,1,(D,N))
    M_est = np.dot(W_est,C_est)

    #Plot the W_ests
    ims = []
    plt.figure(1)
    for i in range(N):
        ax = plt.subplot2grid((N,3),(i,2))
        im = plt.imshow(W_est[:,i].T[:,None],interpolation='none',
        aspect=3./D,cmap=colormap,vmin=0,vmax=1)
        pltuls.strip_bare(ax,axis='x')
        pltuls.strip_ticks(ax,axis='y')
        ims.append(im)
    plt.text(0.65,0.95,'Estim. W',transform=plt.gcf().transFigure)

    #Bar plot of W and W_est
    bar_width = 0.35
    barcollections = []
    for i in range(N):
        ax = plt.subplot2grid((N,3),(i,1))
        plt.bar(np.arange(D),util.normalize(W[:,i]),bar_width,color='r')
        barcollection = plt.bar(np.arange(D)+bar_width,util.normalize(W_est[:,i]),bar_width,color='b')
        plt.ylim([0,1.5])
        pltuls.strip_bare(ax,axis='y')
        barcollections.append(barcollection)


    #Plot the collected (noisy) M, and the current estimate of M
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.imshow(M_nsy,interpolation='none',cmap=colormap,vmin=0,vmax=1)
    plt.title('True M')
    plt.subplot(2,1,2)
    im1 = plt.imshow(M_est,interpolation='none',cmap=colormap,vmin=0,vmax=1)
    plt.title('Estim. M')

    #Show plots thus far
    plt.show()

    error = substeps.compute_error_by_trace(M_nsy,W_est,C_est) #Initial error computation

    counter = 1
    while error/SS_tot > error_threshold:
        # print('----ITERATION----'+str(counter))
        # print('error: '+str(error/SS_tot))

        #multiplicative update of estimated W and C (d'Avella and Bizzi 2005 supplemental text)
        C_est = substeps.mult_update_c_synchronous(M_nsy,W_est,C_est)
        W_est = substeps.mult_update_W_synchronous(M_nsy,W_est,C_est)
        if counter%25==1:
            #Match estimated Ws to true Ws
            true_syn_partners = substeps.match_synergy_estimates_sync(W,W_est)

            #update W_est image and bar plot, M_est image
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

            plt.pause(0.00001)

        #Recompute error
        error = substeps.compute_error_by_trace(M_nsy,W_est,C_est)
        counter+=1

    #-----Similarity metrics for this particular algorithm iteration

    #(1) the subspace similarities
    angles = scipy.linalg.subspace_angles(W,W_est)
    #Compute the average of the cosines of the principal angles
    subspace_similarities_b_s[1,k] = np.mean(np.cos(angles))
    #Compute the d_b for principal angles
    C_b = np.random.exponential(1./lam,(num_reps-1,N,T))
    W_b = np.random.exponential(1./lam,(num_reps-1,D,N))
    d_bs_cos = np.zeros(num_reps-1)
    for j in range(num_reps-1):
        angles = scipy.linalg.subspace_angles(W_b[j,:,:],W_est)
        d_bs_cos[j] = np.mean(np.cos(angles))
    subspace_similarities_b_s[0,k] = np.mean(d_bs_cos)

    #(2) the basis vector similarities
    basis_vector_similarities_b_s[1,k] = np.mean(
        [np.corrcoef(W[:,i],W_est[:,true_syn_partners[i]]) for i in range(N)])
    d_bs_vectors = np.zeros(num_reps-1)
    for j in range(num_reps-1):
        d_bs_vectors[j] = np.mean(
        [np.corrcoef(W_b[j,:,i],W[:,true_syn_partners[i]]) for i in range(N)])
    basis_vector_similarities_b_s[0,k] = np.mean(d_bs_vectors)

    #(3) the coefficent similarities
    coefficient_similarities_b_s[1,k]  = np.mean(
        [np.corrcoef(C[i,:],C_est[true_syn_partners[i],:]) for i in range(N)])
    d_bs_coeffs = np.zeros(num_reps-1)
    for j in range(num_reps-1):
        d_bs_coeffs[j]= np.mean(
        [np.corrcoef(C_b[j,i,:],C_est[true_syn_partners[i],:]) for i in range(N)])
    coefficient_similarities_b_s[0,k] = np.mean(d_bs_coeffs)


#Now merge everything accross all algorithm repetitions
#Compute overall subspace similarity
subspace_similarities_d_b = np.mean(subspace_similarities_b_s[0,:])
subspace_similarities_d_s = np.mean(subspace_similarities_b_s[1,:])
subspace_similarity = \
    (subspace_similarities_d_s-subspace_similarities_d_b)/(
    1. - subspace_similarities_d_b)
print('subspace similarity: '+str(subspace_similarity))

#Compute overall basis vector similarity
basis_vector_similarities_d_b=np.mean(basis_vector_similarities_b_s[0,:])
basis_vector_similarities_d_s=np.mean(basis_vector_similarities_b_s[1,:])
basis_vector_similarity = \
    (basis_vector_similarities_d_s-basis_vector_similarities_d_b)/(
    1. - basis_vector_similarities_d_b)
print('basis vector similarity: '+str(basis_vector_similarity))

#Compute overall coefficient similarity
coefficient_similarities_d_b=np.mean(coefficient_similarities_b_s[0,:])
coefficient_similarities_d_s=np.mean(coefficient_similarities_b_s[1,:])
coefficient_similarity = \
    (coefficient_similarities_d_s-coefficient_similarities_d_b)/(
    1. - coefficient_similarities_d_b)
print('coefficient similarity: '+str(coefficient_similarity))

raw_input('here')
