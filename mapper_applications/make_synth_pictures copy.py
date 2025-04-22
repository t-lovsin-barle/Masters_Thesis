# Code to recreate results of Mapper applied to concentric circles

import sys
import os

# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)

import numpy as np
import math
import gtda.mapper as mpr  # type: ignore
from sklearn.datasets import make_circles  # type: ignore

from automato import Automato
from dataset_utils import plotting
from scipy.spatial.distance import directed_hausdorff


# Create concentric circles
n = 5000
X, y = make_circles(n_samples=n, noise=0.05, factor=0.3, random_state=42)
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

# Choose m points randomly
beta = 1.1
s_n = round(n / (np.log(n) ** beta)) # Sample size
indices = np.random.choice(len(X), size=s_n, replace=False)
X_sampled = X[indices]
y_sampled = y[indices]

# Hausdorff distance between X and X_sampled
d_XXs = directed_hausdorff(X,X_sampled)[0]
d_XsX = directed_hausdorff(X_sampled,X)[0]
hausdorff_dist = max(d_XXs,d_XsX)

# Instantiate Mapper
filter_func = mpr.Projection(columns=[0])  # Specify filter
#delta1 = hausdorff_dist
delta2 = 8 * np.sqrt((2 * np.log(n) / n))
overlap_frac = 0.4  # Specify fractional overlap
n_intervals = math.ceil(2 / (delta2 / overlap_frac))  # Specify numbers of intervals to use

cover = mpr.CubicalCover(
    n_intervals=n_intervals,
    overlap_frac=overlap_frac
)

n_jobs = 1
clusterer = Automato(tomato_params={'graph_type':'radius','r':delta2,'density_type':'DTM'})
pipe = mpr.make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=False,
    n_jobs=n_jobs, 
    min_intersection=1
    )
    # Create Mapper graph
mapper_graph = pipe.fit_transform(X)
    # Create Mapper figure
plotly_params = {
    "node_trace": {
    "marker_colorscale": "viridis",
    "marker_showscale": False
    }
}
fig = mpr.plot_static_mapper_graph(
    pipe,
    X,
    color_data=y,
    layout_dim=2,
    plotly_params=plotly_params
)
fig.update_xaxes(
    showline=True,
    linewidth=1,
    linecolor='black',
    mirror=True
)
fig.update_yaxes(
    showline=True,
    linewidth=1,
    linecolor='black',
    mirror=True
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
    "./mapper_applications/figures/mapper_concentric_circles_"
    + "DTMbeta1.1_intervals_"
    + "custom_overlap.svg"
)
fig.write_image(filename)
