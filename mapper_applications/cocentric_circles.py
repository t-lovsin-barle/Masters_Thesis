# Code to recreate results of Mapper applied to COIL data

import sys
import os
 
# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)

import gtda.mapper as mpr  # type: ignore
from custom_cover import ResolutionCover
from custom_clusterer import RipsClustering
import numpy as np
import sklearn

from automato import Automato
from helper_functions import compute_parameters, DTM_cuttoff
from sklearn.datasets import make_circles

from dataset_utils import plotting


noise_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

if not os.path.exists("./mapper_applications/figures_circles"):
    os.mkdir("./mapper_applications/figures_circles")

for noise in noise_levels:
    if not os.path.exists("./mapper_applications/figures_circles/noise_" + f"{noise}"):
        os.mkdir("./mapper_applications/figures_circles/noise_" + f"{noise}")
    
    X, y = make_circles(n_samples=1000, noise=noise, factor=0.3, random_state=42)
    fig = plotting.plot_point_cloud(
                X,
                labels = y,
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
            
    filename = ("./mapper_applications/figures_circles/noise_" 
                + f"{noise}/circles_{noise}.svg")
    fig.write_image(filename)

    overlap_fracs = [0.35, 0.4, 0.45, 0.49]
    filters = [
            mpr.Projection(columns=1),
            mpr.Eccentricity(exponent=np.inf),
            sklearn.decomposition.PCA(n_components=1)
    ]
    filter_names = [
            "ProjectionY",
            "Eccentricity",
            "PCA"
    ]
    
    for overlap_frac in overlap_fracs:
        for filter_func , filter_name in zip(filters, filter_names):
            delta, resolution, k = compute_parameters(X=X,
                                                filter_func=filter_func.fit(X).transform(X).flatten(),
                                                gain=overlap_frac)
            print(f"Noise: {noise} \nFilter: {filter_name} \nGain: {overlap_frac} \nRips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}")
            clusterers = [
                Automato(random_state=42),
                Automato(tomato_params={'k_DTM':k,
                                        'graph_type':'radius', 
                                        'r':delta,
                                        'q':2}, 
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
                plotly_params = {"node_trace": {"marker_colorscale": "Viridis"}}
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

                filename = (
                    "./mapper_applications/figures_circles/noise_" 
                    + f"{noise}/circles_{noise}_{filter_name}_{clusterer_name}_{overlap_frac}.svg"
                    )
                fig.write_image(filename)