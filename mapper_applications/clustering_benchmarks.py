import clustbench

import sys
import os

import sklearn.decomposition
 
# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)
data_path = r"C:\Users\trist\Downloads\clustering-data-v1-1.1.0\clustering-data-v1-1.1.0"
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

from helper_functions import DTM_cuttoff, compute_parameters

import numpy as np
import pandas as pd

battery_names = ["wut", "sipu", "fcps", "other"]

battery_sets = {
    "wut": ["circles", "graph", "labirynth", "mk2", "olympic", "stripes", "z1", "z3"],
    "sipu": ["compound"],
    "fcps": ["chainlink", "target"],
    "other": ["iris", "square"]
}

'''
https://archive.ics.uci.edu/dataset/39/ecoli
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
https://archive.ics.uci.edu/dataset/109/wine
'''
uci_sets = ["ecoli", "wdbc", "wine"]





for battery in battery_names:
    if not os.path.exists("./mapper_applications/figures_" + f"{battery}"):
            os.mkdir("./mapper_applications/figures_" + f"{battery}")
    for set in battery_sets[battery]:
        if not os.path.exists("./mapper_applications/figures_" + f"{battery}/{set}"):
            os.mkdir("./mapper_applications/figures_" + f"{battery}/{set}")
        data_set = clustbench.load_dataset(battery,set,path=data_path)
        X = data_set.data
        y = data_set.labels[0]
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
                print(f"Battery: {battery} \nSet: {set} \nFilter: {filter_name} \nGain: {overlap_frac} \nRips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}")
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
                        "./mapper_applications/figures_" 
                        + f"{battery}/{set}/{set}_{filter_name}_{clusterer_name}_{overlap_frac}.svg"
                    )
                    fig.write_image(filename)