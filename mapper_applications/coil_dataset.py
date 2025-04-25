# Code to recreate results of Mapper applied to COIL data

import os

import gtda.mapper as mpr  # type: ignore
from custom_cover import ResolutionCover
from custom_clusterer import AutoRipsClusering
import numpy as np
import pandas as pd  # type: ignore
import math
import sklearn

from automato import Automato
from scipy.spatial.distance import pdist, squareform
from helper_functions import average_delta, empiric_mod_of_contin

import numpy as np
import pandas as pd
from PIL import Image


# Path to the image folder
image_dir = r'C:\Users\trist\Downloads\coil-100\coil-100'

data = []

# Loop through obj74 images
for fname in os.listdir(image_dir):
    if fname.startswith("obj74__") and fname.endswith(".png"):
        angle = int(fname.split('__')[1].split('.')[0])
        img_path = os.path.join(image_dir, fname)

        # Load image, convert to grayscale and scale pixel values to [0, 1]
        img = Image.open(img_path).convert('L')  # 'L' for grayscale
        img_array = np.array(img) / 255.0  # normalize to [0, 1]
        img_flat = img_array.flatten()

        # Compute label
        label = 1 - abs(angle - 180) / 180

        # Add to data
        data.append(np.append(img_flat, label))

# Create DataFrame
df = pd.DataFrame(data)
X, y = df.values[:, :-1], df.values[:, -1]

# Instantiate Mapper parameters
overlap_frac = 0.4  # Specify fractional overlap (gain)
V = empiric_mod_of_contin(
    func=sklearn.decomposition.PCA(n_components=1).fit(X).transform(X), 
    delta=average_delta(X),
    dist_mtrx= squareform(pdist(X, metric = 'euclidean'))
    )
resolution = V[0] / overlap_frac

clusterer = AutoRipsClusering()
#clusterer = Automato(random_state=42)
#clusterer = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)
cover = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
n_jobs = 1
pipe_custom = mpr.make_mapper_pipeline(
    filter_func=sklearn.decomposition.PCA(n_components=1),
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
    "./mapper_applications/figures/Coil_dataset_Bruh.svg"
)
fig.write_image(filename)
