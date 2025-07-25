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

from automato import Automato
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from helper_functions import compute_parameters, DTM_cuttoff

import numpy as np
import pandas as pd



# Create DataFrame
df = pd.read_csv("./mapper_applications/player_stats_traditionnal_rs.csv", index_col=[0])

# Filter for the 2011-12 season
df_filtered = df[df['SEASON'] == '2011-12']

# Columns of interest
stats_cols = ['REB', 'AST', 'TOV', 'STL', 'BLK', 'PTS']

# Avoid division by zero or missing values
df_filtered = df_filtered[df_filtered['MIN'] > 0]

# Divide each stat column by MIN
per_minute_stats = df_filtered[stats_cols].div(df_filtered['MIN'], axis=0)

X= per_minute_stats.values[:, :-1]

# Compute label from raw PTS (not per-minute)
pts_raw = df_filtered['PTS']
pts_min = pts_raw.min()
pts_max = pts_raw.max()
y = ((pts_raw - pts_min) / (pts_max - pts_min)).values
X = StandardScaler().fit_transform(X)
denoisify = True

if denoisify:
    mask = DTM_cuttoff(X)
    X, y = X[mask], y[mask]

filter_func=sklearn.decomposition.TruncatedSVD(n_components=1)
overlap_frac = 0.4

delta, resolution, k = compute_parameters(X=X,
                                          filter_func=filter_func.fit(X).transform(X),
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
        "./mapper_applications/figures_new/NBA_"
        + f"{clusterer_name}_{overlap_frac}"
        + "_gain_denoised.svg"
    )
    fig.write_image(filename)
