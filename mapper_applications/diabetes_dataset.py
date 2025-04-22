# Code to recreate results of Mapper applied to diabetes data

import sys
import os

# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)


import gtda.mapper as mpr  # type: ignore
from gtda.mapper.cover import CubicalCover
import gudhi
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import math

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed
from scipy.spatial.distance import pdist, squareform, directed_hausdorff

# Load diabetes data
df = pd.read_csv("./mapper_applications/chemdiab.csv", index_col=[0])
X, y = df.values[:, :-1], df.values[:, -1]
X = StandardScaler().fit_transform(X)
y = np.where(y == "Normal", 0, y)
y = np.where(y == "Chemical_Diabetic", 1, y)
y = np.where(y == "Overt_Diabetic", 2, y)

print(f"size of Data is {len(X)}")

def average_delta(data, n_iterations, beta): # Average delta over N random samples
    delta = 0
    s_n = math.ceil(len(data) / (np.log(len(data)) ** (1 + beta))) # Sample size
    print
    for i in range(n_iterations):

        # Choose m points randomly
        indices = np.random.choice(len(data), size=s_n, replace=False)
        data_sampled = data[indices]

        # Hausdorff distance between data and data_sampled
        d_XXs = directed_hausdorff(data,data_sampled)[0]
        d_XsX = directed_hausdorff(data_sampled,data)[0]
        delta = delta + max(d_XXs,d_XsX)
    print(f"Sample size is {s_n}")
    return delta / n_iterations

def empiric_mod_of_contin(func, delta, dist_mtrx, epsilon):
    V = 0
    for i in range(len(dist_mtrx)):
        for j in range(i,len(dist_mtrx)):
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

# Create Rips complex
rips_complex = gudhi.RipsComplex(
    points = X,
    max_edge_length = delta
).create_simplex_tree(max_dimension = 1)

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
print(f"image length is {np.max(Xt)-np.min(Xt)}")
n_intervals = math.ceil((np.max(Xt)-np.min(Xt)) / resolution) + 2 # Specify numbers of intervals to use
print(f"Resolution is {resolution} and number of intervals is {n_intervals}")

#clusterer = mpr.FirstSimpleGap(relative_gap_size=relative_gap_size)
clusterer = Automato(random_state=42)

cover = mpr.CubicalCover(
    n_intervals=n_intervals,
    overlap_frac=overlap_frac
)
#print(f"cover: \n", cover.fit_transform(X))
n_jobs = 1
pipe_custom = mpr.make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
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
    "./mapper_applications/figures/diabetes_dataset_baseline.svg"
)
fig.write_image(filename)
