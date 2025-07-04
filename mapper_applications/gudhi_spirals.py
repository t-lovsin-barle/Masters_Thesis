# Code to recreate results of Mapper applied to COIL data

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
import sklearn
import math
from dataset_utils import plotting
from scipy.spatial.distance import cdist

from automato import Automato
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from helper_functions import average_delta, empiric_mod_of_contin
from sklearn.datasets import make_circles

import numpy as np
import pandas as pd
import gudhi.datasets.remote

X = gudhi.datasets.remote.fetch_spiral_2d(file_path=None)
print(X)

z = X[:, 1]

# Normalize z to range [0, 1]
z_min = z.min()
z_max = z.max()
y = (z - z_min) / (z_max - z_min)

print(y)

delta = average_delta(X,n_iterations=10)
print(delta)
# Instantiate Mapper parameters 
overlap_frac = 0.3  # Specify fractional overlap (gain)
V, k = empiric_mod_of_contin(
    func=mpr.Projection(columns=[1]).fit(X).transform(X), 
    delta=delta,
    dist_mtrx= cdist(X,X)
    )
resolution = V[0] / overlap_frac 
print(resolution)
print(k)

clusterer1 = RipsClustering(max_edge_length=delta)
clusterer3 = Automato(random_state=42)
clusterer2 = Automato(tomato_params={'k_DTM':k,'graph_type':'radius', 'r':delta}, random_state=42)
clusterer4 = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)
cover = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
n_jobs = 1


pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[1]),
    cover=cover,
    clusterer=clusterer2,
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
    "./mapper_applications/figures/bunny_AutomatoRipsK.svg"
)
fig.write_image(filename)

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[1]),
    cover=cover,
    clusterer=clusterer1,
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
    "./mapper_applications/figures/spirals_Rips.svg"
)
fig.write_image(filename)

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[1]),
    cover=cover,
    clusterer=clusterer3,
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
    "./mapper_applications/figures/spirals_Automato.svg"
)
fig.write_image(filename)

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[1]),
    cover=cover,
    clusterer=clusterer4,
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
    "./mapper_applications/figures/spirals_AutomatoRong.svg"
)
fig.write_image(filename)
