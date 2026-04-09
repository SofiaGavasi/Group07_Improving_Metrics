from __future__ import annotations

from typing import Any, Dict
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances


# assume shape of real samples is [N_real,D] (check if otherwise), numpy array 
def compute_precision_recall(real_samples: Any, fake_samples: Any, k=3,**kwargs: Any):
  

    # start with the steps 


    # add some checks for dimensions (later)

    N_real,_  = real_samples.shape
    N_fake,_ = fake_samples.shape



    # nearest neighbours distances 




    # k-nearest neigbour for each sample in real set 
    all_distances_real_real = pairwise_distances(real_samples, real_samples) # first we compute the distance between real samples
    real_k_nearest_neighbours = nearest_neighbour_computation(k,all_distances_real_real,N_real) # k nearest neigbour per sample

 

    # k-nearest neigbour for each sample in generated set 
    all_distances_fake_fake = pairwise_distances(fake_samples, fake_samples)
    fake_k_nearest_neighbours = nearest_neighbour_computation(k,all_distances_fake_fake,N_fake)


   
    # we need real-generated samples euclidean distances for the precision and recall calculations
    all_distances_real_fake = pairwise_distances(real_samples, fake_samples)


   
   
    real_k_nearest_neighbours = real_k_nearest_neighbours.reshape(-1,1)
    fake_covered = np.any(all_distances_real_fake <= real_k_nearest_neighbours, axis=0)
    precision = np.mean(fake_covered)


    fake_k_nearest_neighbours = fake_k_nearest_neighbours.reshape(1,-1)
    real_covered = np.any(all_distances_real_fake <= fake_k_nearest_neighbours, axis=1)
    recall = np.mean(real_covered)







    #  return keys: {"precision": float, "recall": float}
    return {"precision": precision, "recall" : recall}




def  nearest_neighbour_computation(k,pairwise_dists,num_samples):
    
    
    k_nearest_neighbours = np.zeros(num_samples) # the k-nearest neigbour of each real sample 
    for i in range(num_samples):
        distances_sample  = pairwise_dists[i].copy()
        distances_sample[i] = np.inf # making sure that the self distance is excluded from the neighbours 
        distances_sample.sort()  
        k_nearest_neighbours[i] = distances_sample[k - 1]

    return k_nearest_neighbours




 # previous code to calculate distances between pairs real-real, fake-fake, real-fake (less efficient)

    # # could do a helper function

    # # the k-nearest neighbour per real data sample
    # real_k_nearest_neigbours = np.zeros(N_real)

    # for i in range(N_real):
    #     all_distances = [] # try to do these w/out a list

    #     for j in range(N_real):
    #         if i != j:
    #             distance = np.linalg.norm(real_samples[i] - real_samples[j])
    #             all_distances.append(distance)






    #     all_distances.sort() # to find the k-th position we sort by increasing distance 
    #     sample_k_nearest = all_distances[k - 1]
    #     real_k_nearest_neigbours[i] = sample_k_nearest    



    # # same for generated

    # # helper

    # fake_k_nearest_neigbours = np.zeros(N_fake)

    # for i in range(N_fake):
    #     all_distances = []

    #     for j in range(N_fake):
    #         if i != j:
    #             distance = np.linalg.norm(fake_samples[i] - fake_samples[j]) 
    #             all_distances.append(distance)

    #     all_distances.sort()
    #     sample_k_nearest = all_distances[k - 1]
    #     fake_k_nearest_neigbours[i] = sample_k_nearest  




    # all_distances_real_fake = np.zeros((N_real,N_fake))

    # for i in range(N_real):
    #     for j in range(N_fake):
    #         distance = np.linalg.norm(real_samples[i] - fake_samples[j])
    #         all_distances_real_fake[i][j] = distances



