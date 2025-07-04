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

X, y = make_circles(n_samples=2000, noise=0, factor=0.3, random_state=42)
fig = plotting.plot_point_cloud(
    X,
    labels=y,
    to_scale=True
)
fig.update_xaxes(
        title="x",
        tickmode='linear'
)
fig.update_yaxes(
        title="y",
        tickmode='linear'
)
fig.update_layout(
    autosize=False,
    width=400,
    height=400,
)
if not os.path.exists("./mapper_applications/figures/"):
    os.mkdir("./mapper_applications/figures/")
filename = "./mapper_applications/figures/concentric_circles.svg"
fig.write_image(filename)

delta = average_delta(X)
print(delta)
# Instantiate Mapper parameters
overlap_frac = 0.3  # Specify fractional overlap (gain)
V, k = empiric_mod_of_contin(
    func=mpr.Projection(columns=[0]).fit(X).transform(X), 
    delta=delta,
    dist_mtrx= cdist(X,X)
    )
resolution = V[0] / overlap_frac 

print(k)

clusterer1 = RipsClustering(max_edge_length=delta)
clusterer3 = Automato(random_state=42)
clusterer2 = Automato(tomato_params={'k_DTM':k,'graph_type':'radius', 'r':delta}, random_state=42)
cover = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
n_jobs = 1

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[0]),
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
    "./mapper_applications/figures/cocentric_circles_Rips.svg"
)
fig.write_image(filename)

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[0]),
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
    "./mapper_applications/figures/cocentric_circles_AutomatoRipsk.svg"
)
fig.write_image(filename)

pipe_custom = mpr.make_mapper_pipeline(
    filter_func=mpr.Projection(columns=[0]),
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
    "./mapper_applications/figures/cocentric_circles_Automato.svg"
)
fig.write_image(filename)
