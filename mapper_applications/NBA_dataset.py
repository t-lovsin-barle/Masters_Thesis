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
from helper_functions import average_delta, empiric_mod_of_contin

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


delta = average_delta(X)
# Instantiate Mapper parameters
overlap_frac = 0.49  # Specify fractional overlap (gain)
V = empiric_mod_of_contin(
    func=sklearn.decomposition.TruncatedSVD(n_components=1).fit(X).transform(X), 
    delta=delta,
    dist_mtrx= squareform(pdist(X, metric = 'euclidean'))
    )
resolution = V / overlap_frac

n_intervals = (max(sklearn.decomposition.TruncatedSVD(n_components=1).fit(X).transform(X))-min(sklearn.decomposition.TruncatedSVD(n_components=1).fit(X).transform(X)))/resolution
print(n_intervals)
clusterer = RipsClustering(max_edge_length=delta)
#clusterer = Automato(random_state=42)
#clusterer = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)
cover = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
n_jobs = 1
pipe_custom = mpr.make_mapper_pipeline(
    filter_func=sklearn.decomposition.TruncatedSVD(n_components=1),
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
    "./mapper_applications/figures/NBA_dataset_Rips.svg"
)
fig.write_image(filename)
