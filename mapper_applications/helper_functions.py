# Helper functions to be used for computing parameters for Mapper. Banished here in order to make other scripts more readable
import numpy as np
import math

import sys
import os
 
# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)

from scipy.spatial.distance import cdist
from gudhi.point_cloud.dtm import DTMDensity

from automato import Automato
# Average delta over N random samples
def average_delta(X, dist_matrix=None, n_iterations=100, beta=0.001, magic = 10): 
    delta = 0
    s_n = math.floor(len(X) / (np.log(len(X)) ** (1 + beta)) * np.log(magic) ** (1 + beta))
    if dist_matrix is None:
        dist_matrix = cdist(X,X)
    for i in range(n_iterations):
        indices = np.random.choice(len(X), size=s_n, replace=False)
        sampled_dist_matrix=dist_matrix[:,indices]
        hausdorff_dist_from_data_to_sample = 0

        min_j = 0
        for j in range(len(X)):
            min_j = min(sampled_dist_matrix[j,:])

            hausdorff_dist_from_data_to_sample = max(hausdorff_dist_from_data_to_sample, min_j)
        delta = delta + hausdorff_dist_from_data_to_sample / n_iterations
    return delta

# Computation of the V(delta)
def empiric_mod_of_contin(filter_func, delta, dist_matrix, epsilon=0.001): 
    V = 0
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if dist_matrix[i,j] <= delta:
                V_ = abs(filter_func[i]-filter_func[j])
                if V <= V_:
                    V = V_

    return float(V + epsilon)

def optimal_resolution(mod_of_contin, gain = 0.4):
    resolution = mod_of_contin / gain
    return resolution

def compute_parameters(X,
                       n_iterations=100, 
                       beta=0.001, 
                       magic=10,
                       filter_func=None,
                       dist_matrix=None,
                       epsilon=0.001,
                       gain=0.4):

    if dist_matrix is None:
        dist_matrix = cdist(X,X)

    delta = average_delta(X=X, 
                          dist_matrix=dist_matrix, 
                          n_iterations=n_iterations, 
                          beta=beta, 
                          magic=magic)
    mod_of_contin = empiric_mod_of_contin(filter_func=filter_func,
                                          delta=delta,
                                          dist_matrix=dist_matrix,
                                          epsilon=epsilon)
    resolution = optimal_resolution(mod_of_contin=mod_of_contin,
                                    gain=gain)
    DTM_parameter = int(np.ceil(len(X) * 0.05))
    return delta, resolution, DTM_parameter

def DTM_cuttoff(X, m=0.05, percentile = 0.05):
    k=int(np.ceil(len(X)*m))
    estimated_density = DTMDensity(k=k,q=2).fit(X).transform(X)
    sorted_density = np.sort(estimated_density)[::-1]
    top_percentile = int(len(estimated_density) * percentile)
    threshold = sorted_density[top_percentile - 1]
    mask = estimated_density < threshold
    return mask


def denoiser(X, n_features = False, lifespan_sum = True):
    m = 0.05
    criterion_value = -np.inf
    labels = np.ones(len(X), dtype=bool)

    if n_features == True:
        for i in range(5):
            m_ = 0.05 * (i+1)
            k = int(np.ceil(len(X)*m_))
            automato = Automato(tomato_params={'k':k}).fit(X)
            criterion_value_temp = n_prominent_features(diagram=automato.diagram_,
                                            quantile=automato.width_conf_band_)
            if criterion_value_temp > criterion_value:
                criterion_value = criterion_value_temp
                m = m_
                labels = automato.labels_
    elif lifespan_sum == True:
        for i in range(5):
            m_ = 0.05 * (i+1)
            k = int(np.ceil(len(X)*m_))
            automato = Automato(tomato_params={'k':k}, random_state=42).fit(X)
            criterion_value_temp = prominent_lifespan_sum(diagram=automato.diagram_,
                                            quantile=automato.width_conf_band_)
            if criterion_value_temp > criterion_value:
                criterion_value = criterion_value_temp
                m = m_
                labels = automato.labels_
    else:
        raise ValueError("At least one of n_features or lifespan_sum must be True.")

    return m, labels

def prominent_lifespan_sum(diagram, quantile):
    sum = 0
    for feature in range(len(diagram)):
        summand = diagram[feature,0] - diagram[feature,1] - quantile/2
        if summand == float('inf'):
            summand = 0
        if  summand > 0:
            sum = sum + summand
    return sum

def n_prominent_features(diagram, quantile):
    num = 0
    for feature in range(len(diagram)):
        feature_lifespan = diagram[feature,0] - diagram[feature,1]
        if feature_lifespan > quantile/2:
            num = num + 1
    return num