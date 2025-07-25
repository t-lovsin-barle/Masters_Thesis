# Code to recreate results of Mapper applied to COIL data

import sys
import os

import sklearn.decomposition
 
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
from sklearn.preprocessing import Normalizer

from automato import Automato
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from helper_functions import DTM_cuttoff, compute_parameters
from sklearn.datasets import make_circles
from gudhi.datasets.generators import points

import numpy as np
import pandas as pd
from mapper_applications.custom_filters import Norm

df = pd.read_csv("./mapper_applications/simon_TDA_sphere_with_cylinders.csv", index_col=[0])
X = df.values[:, :-1]

filter_func=sklearn.decomposition.PCA(n_components=1)
func =filter_func.fit(X).transform(X).flatten()
y= (func-min(func))/(max(func)-min(func))

overlap_frac = 0.4

denoisify = False

if denoisify:
    mask = DTM_cuttoff(X)
    X, y = X[mask], y[mask]

delta, resolution, k = compute_parameters(X=X,
                                          filter_func=filter_func.fit(X).transform(X).flatten(),
                                          gain=overlap_frac)
print(f"Rips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}")
clusterers = [
    Automato(random_state=42),
    Automato(tomato_params={'k_DTM':k,
                            'graph_type':'radius', 
                            'r':delta}, 
            random_state=42),
    RipsClustering(max_edge_length=delta)
]
clusterer_names= [
    "automato",
    "tuned_auto",
    "rips_clusers"
]
cover = ResolutionCover(
        resolution=resolution,
        gain=overlap_frac
    )
for clusterer, clusterer_name in zip(clusterers, clusterer_names):
    
    n_jobs = 1

    # Create Mapper pipeline
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
        node_scale=10
    )
    fig.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    # Save Mapper figure to disk
    if not os.path.exists("./mapper_applications/figures_new/"):
        os.mkdir("./mapper_applications/figures_new/")
    filename = (
        "./mapper_applications/figures_new/TDA2PCA_"
        + f"{clusterer_name}_{overlap_frac}"
        + "_gain.svg"
    )
    fig.write_image(filename)