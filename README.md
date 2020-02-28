# Muscle Synergy Extraction


The theory of "muscle synergies" posits that the muscle activity we observe in a set of muscles is some linear combination of characteristic
fixed-relationship muscle time courses "synergies" (see
[here](https://papers.nips.cc/paper/1974-modularity-in-the-motor-system-decomposition-of-muscle-patterns-as-combinations-of-time-varying-synergies.pdf) for an overview).   
Goal: Given a muscle activity time course that is a combination of synergies, can we reconstruct the individual synergies and their coefficients?  
Here we implement this reconstruction using non-negative matrix factorization as described [here](https://www.researchgate.net/publication/10921359_Combinations_of_muscle_synergies_in_the_construction_of_a_natural_motor_behavior).  
The animation below demonstrates the successful convergence of our implementation. We give as input a linear combination of the some randomly chosen synergy patterns (right), initialize with completely random synergies (left), and show that we can recover the synergies we started with.  
![Alt Text](https://media.giphy.com/media/ZcRNsz54Bbvj3xwhpa/giphy.gif)
