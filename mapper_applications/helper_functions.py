# Helper functions to be used for computing parameters for Mapper. Banished here in order to make other scripts more readable
import numpy as np
import math

from scipy.spatial.distance import cdist

# Average delta over N random samples
def average_delta(X, n_iterations=100, beta=0.001): 
    delta = 0
    s_n = math.floor(len(X) / (np.log(len(X)) ** (1 + beta)))
    for i in range(n_iterations):
        indices = np.random.choice(len(X), size=s_n, replace=False)
        all_indices = np.arange(len(X))
        c_indices = np.setdiff1d(all_indices, indices)
        dist_matrix = cdist(X[c_indices],X[indices])
        hausdorff_dist_from_data_to_sample = 0

        for j in range(len(c_indices)):
            min_j = dist_matrix[j,0]

            for k in range(len(indices)):
                min_j = min(min_j, dist_matrix[j,k])

        hausdorff_dist_from_data_to_sample = max(hausdorff_dist_from_data_to_sample, min_j)
        delta = delta + hausdorff_dist_from_data_to_sample / n_iterations
    return delta

# Computation of the V(delta)
def empiric_mod_of_contin(func, delta, dist_mtrx, epsilon=0.001): 
    V = 0
    for i in range(len(dist_mtrx)):
        for j in range(len(dist_mtrx)):
            if dist_mtrx[i,j] <= delta:
                V_ = abs(func[i]-func[j])
                if V <= V_:
                    V = V_
    return V + epsilon