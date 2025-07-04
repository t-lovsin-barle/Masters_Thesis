# Code to recreate results of Mapper applied to diabetes data
import sys
import os
 
# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)

import gtda.mapper as mpr  # type: ignore
from custom_cover import ResolutionCover
from custom_clusterer import RipsClustering, AutoRipsClustering
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import math
from helper_functions import average_delta, empiric_mod_of_contin

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed
from scipy.spatial.distance import pdist, squareform

# Load diabetes data
df = pd.read_csv("./mapper_applications/chemdiab.csv", index_col=[0])
X, y = df.values[:, :-1], df.values[:, -1]
X = StandardScaler().fit_transform(X)
y = np.where(y == "Normal", 0, y)
y = np.where(y == "Chemical_Diabetic", 1, y)
y = np.where(y == "Overt_Diabetic", 2, y)

# Instantiate Mapper parameters
filter_func = EccentricitySubclassed(exponent=np.inf)  # Specify filter
overlap_frac = 0.4  # Specify fractional overlap (gain)
delta = average_delta(X)
V = empiric_mod_of_contin(
    func=mpr.Eccentricity(exponent=np.inf).fit(X).transform(X), 
    delta=delta,
    dist_mtrx=squareform(pdist(X, metric = 'euclidean'))
    )
resolution = V[0] / overlap_frac
#clusterer = RipsClustering(max_edge_length=delta)
#clusterer= RipsClustering(max_edge_length=average_delta(X))
clusterer = Automato(random_state=42)
#clusterer = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)
cover2 = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
n_jobs = 1

# Create Mapper pipeline
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
    "./mapper_applications/figures/diabetes_dataset_Automato.svg"
)
fig.write_image(filename)
