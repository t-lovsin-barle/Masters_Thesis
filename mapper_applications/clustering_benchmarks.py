'''Code to generate Mappers using synthetic data sets from Clusterin Benchmarks, 
with varying parameters for the cover and Automato algorithm. The 'control group'
are the Mappers using RipsClustering as the clustering algorithm'''

# Clustering benchmarks download folder
import clustbench
data_path = r"C:\Users\trist\Downloads\clustering-data-v1-1.1.0\clustering-data-v1-1.1.0"

# Code to import automato
import sys
import os

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)

# Custom objects import
from custom_cover import ResolutionCover
from custom_clusterer import RipsClustering
from automato import Automato
from helper_functions import compute_parameters

import numpy as np
import pandas as pd
import sklearn.decomposition
from dataset_utils import plotting
import gtda.mapper as mpr  # type: ignore


battery_names = ["wut", "sipu", "fcps", "other"]

battery_sets = {
    "wut": ["circles", "graph", "labirynth", "mk2", "olympic", "stripes", "z1", "z3"],
    "sipu": ["compound"],
    "fcps": ["chainlink", "target"],
    "other": ["iris", "square"]
}

# Iterate through batteries
for battery in battery_names:
    # Create folder
    if not os.path.exists("./mapper_applications/figures_" + f"{battery}"):
            os.mkdir("./mapper_applications/figures_" + f"{battery}")
    
    # Iterate though sets
    for set in battery_sets[battery]:

        # Create folder
        if not os.path.exists("./mapper_applications/figures_" + f"{battery}/{set}"):
            os.mkdir("./mapper_applications/figures_" + f"{battery}/{set}")

        data_set = clustbench.load_dataset(battery,set,path=data_path)
        X = data_set.data
        y = data_set.labels[0]

        # Create figure of the data set if the dimension is 2
        if X.shape[1] == 2:
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
            
            filename = ("./mapper_applications/figures_" + f"{battery}/{set}/{set}.svg")
            fig.write_image(filename)
        
        # Innitiate different overlaps and filters. Filters need to have .fit() and .transform() functions
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

        # Not in the scope of the Masters Thesis
        alphas = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        
        # Varying overlap fractions
        for overlap_frac in overlap_fracs:

            # Varying alpha in the bottleneck bootstrap
            for alpha in alphas:

                # Iterating through filters
                for filter_func , filter_name in zip(filters, filter_names):
                    delta, resolution, k = compute_parameters(X=X,
                                                        filter_func=filter_func.fit(X).transform(X).flatten(),
                                                        gain=overlap_frac)
                    print(f"Battery: {battery} \nSet: {set} \nFilter: {filter_name} \nGain: {overlap_frac} \nRips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}\nAlpha: {alpha}")
                    clusterers = [
                        Automato(random_state=42, alpha=alpha),
                        Automato(tomato_params={'k_DTM':k,
                                                'graph_type':'radius', 
                                                'r':delta,
                                                'q':2}, 
                                random_state=42,
                                alpha=alpha),
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
                            "./mapper_applications/figures_" 
                            + f"{battery}/{set}/{set}_{filter_name}_{clusterer_name}_{overlap_frac}_{alpha}.svg"
                        )
                        fig.write_image(filename)