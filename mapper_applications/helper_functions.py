# Code to recreate results of Mapper applied to diabetes data

import sys
import os

# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)


import gtda.mapper as mpr  # type: ignore
from custom_cover import ResolutionCover
from custom_clusterer import AutoRipsClusering
import gudhi
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import math

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed
from scipy.spatial.distance import pdist, squareform, directed_hausdorff, cdist

# Load diabetes data
df = pd.read_csv("./mapper_applications/chemdiab.csv", index_col=[0])
X, y = df.values[:, :-1], df.values[:, -1]
X = StandardScaler().fit_transform(X)
y = np.where(y == "Normal", 0, y)
y = np.where(y == "Chemical_Diabetic", 1, y)
y = np.where(y == "Overt_Diabetic", 2, y) # change from the original to get a 2 color mapper result as in the paper

print(f"size of Data is {len(X)}")

def hausdorff_distance(A, B):
    """
    Computes the directed Hausdorff distance between two point sets A and B in N-dimensional space.
    """
    # Compute pairwise distances between points in A and points in B
    dist_A_to_B = cdist(A, B)
    
    # For each point in A, find the minimum distance to any point in B
    directed_A_to_B = np.max(np.min(dist_A_to_B, axis=1))  # Max of min distances from A to B
    
    # For each point in B, find the minimum distance to any point in A
    directed_B_to_A = np.max(np.min(dist_A_to_B, axis=0))  # Max of min distances from B to A
    
    return directed_A_to_B, directed_B_to_A


def average_delta(X, n_iterations, beta): # Average delta over N random samples
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
def empiric_mod_of_contin(func, delta, dist_mtrx, epsilon):
    V = 0
    for i in range(len(dist_mtrx)):
        for j in range(len(dist_mtrx)):
            if dist_mtrx[i,j] <= delta:
                V_ = abs(func[i]-func[j])
                if V <= V_:
                    V = V_
    return V + epsilon

# Innitiate delta parameters
beta = 0.001
n_iterations = 100
exponent = np.inf

# calculations for relative gap size and eccentricity
dist_mtrx = squareform(pdist(X, metric = 'euclidean'))
Xt = mpr.Eccentricity(exponent=exponent).fit(X).transform(X) # For the calculation of the n_intervals
#print(f"Eccentricity vector:\n", Xt)

np.fill_diagonal(dist_mtrx, np.inf)
row_minima = np.min(dist_mtrx, axis = 1, keepdims=True) # For the calculation of relative_gap_size


# Instantiate Mapper parameters
filter_func = EccentricitySubclassed(exponent=exponent)  # Specify filter
overlap_frac = 0.4  # Specify fractional overlap (gain)
delta = average_delta(X, n_iterations, beta)
print(f"Delta is {delta}")

relative_gap_size = min(1, delta / np.max(Xt)) # parameter for mpr.FirstSimpleGap
print(f"relative gap size is {relative_gap_size}")

epsilon = 0.001
V = empiric_mod_of_contin(
    func=Xt, 
    delta=delta,
    dist_mtrx=dist_mtrx,
    epsilon=epsilon
    )

print(f"V is \n", V)
resolution = V[0] / overlap_frac
print(f"max and min are {np.max(Xt), np.min(Xt)}")
print(f"image length is {np.max(Xt)-np.min(Xt)}")
n_intervals = math.ceil((np.max(Xt)-np.min(Xt)) / resolution) + 2 # Specify numbers of intervals to use
print(f"Resolution is {resolution} and number of intervals is {n_intervals}")

clusterer = AutoRipsClusering()
#clusterer = Automato(random_state=42)
#clusterer = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)

def mock(X, resolution, gain):
    step = resolution * (1 - gain)
    left = np.min(X)
    counter = 1
    while left + resolution <= np.max(X) and np.max(X) + gain * resolution - (left + resolution) >= resolution:
        left += step
        counter += 1
    left += step
    counter += 1
    left_limits = np.linspace(np.min(X), left, num=counter, endpoint=True)
    right_limits = left_limits + resolution

    return left_limits, right_limits
l,r= mock(X=Xt, resolution=resolution, gain=overlap_frac)
print(f"left is {l} and right is {r}")
print("interval lengths", r-l)

cover = mpr.CubicalCover(
    n_intervals=n_intervals,
    overlap_frac=overlap_frac
)
cover2 = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
#print(f"cover: \n", cover.fit_transform(X))
n_jobs = 1
pipe_custom = mpr.make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover2,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs, 
    min_intersection=1
    )
# Create Mapper graph
mapper_graph = pipe_custom.fit(X)

# Create Mapper figure
plotly_params = {"node_trace": {"marker_colorscale": "jet"}}
fig = mpr.plot_static_mapper_graph(
    pipe_custom,
    X,
    color_data=y,
    layout_dim=2,
    plotly_params=plotly_params,
    layout="fruchterman_reingold",
    node_scale=60
)
fig.update_layout(
    autosize=False,
    width=400,
    height=400,
)
# Save Mapper figure to disk
if not os.path.exists("./mapper_applications/figures/"):
    os.mkdir("./mapper_applications/figures/")
filename = (
    "./mapper_applications/figures/diabetes_dataset_RipsGraphH.svg"
)
fig.write_image(filename)
