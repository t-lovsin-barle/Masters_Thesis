# Code to import automato
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd  
import gtda.mapper as mpr  
from sklearn.preprocessing import StandardScaler  

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  
sys.path.insert(0, str(project_root))

# Custom objects import
from core.custom_cover import ResolutionCover
from core.custom_clusterer import RipsClustering
from core.helper_functions import compute_parameters
from external.automato.automato import Automato

# Path definitions
DATA_DIR = project_root / "data"
FIG_DIR = project_root / "figures" / "Diabetes" / "epsilon=0"
# Create figures folder if it doesn't exist
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load diabetes data
df = pd.read_csv("./data/chemdiab.csv", index_col=[0])
X, y = df.values[:, :-1], df.values[:, -1]
X = StandardScaler().fit_transform(X)
y = np.where(y == "Normal", 0, y)
y = np.where(y == "Chemical_Diabetic", 1, y)
y = np.where(y == "Overt_Diabetic", 1, y)
        
# Innitiate different overlaps and filters. Filters need to have .fit() and .transform() functions
overlap_fracs = [0.35, 0.4, 0.45, 0.49]
filters = [
        mpr.Eccentricity(exponent=np.inf)
]
filter_names = [
        "Eccentricity"
]

n_jobs = 1

for overlap_frac in overlap_fracs:
    for filter_func , filter_name in zip(filters, filter_names):
        
        delta, resolution, k = compute_parameters(X=X,
                                            filter_func=filter_func.fit(X).transform(X),
                                            gain=overlap_frac,
                                            epsilon=0)
        print(f"Filter: {filter_name} \nGain: {overlap_frac} \nRips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}")
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
            # Save Mapper figure to disk
            filename = FIG_DIR / f"Diabetes_{filter_name}_{clusterer_name}_{overlap_frac}.svg"
            )
            fig.write_image(filename)

