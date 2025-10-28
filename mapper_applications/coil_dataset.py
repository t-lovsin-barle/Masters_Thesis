'''Code to generate Mappers using the COIL100 as the data set with 
varying parameters for the cover and Automato algorithm. The 'control group'
are the Mappers using RipsClustering as the clustering algorithm'''

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
import sklearn
import pandas as pd
from PIL import Image
import gtda.mapper as mpr  # type: ignore

# Download: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
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
        img_flat = img_array.flatten(order='F')
        
        # Compute label
        label = 1 - abs(angle - 180) / 180

        # Add to data
        data.append(np.append(img_flat, label))

# Create DataFrame
df = pd.DataFrame(data)
X, y = df.values[:, :-1], df.values[:, -1]

# Create folder
if not os.path.exists("./mapper_applications/figures_COIL_epsilon=0"):
    os.mkdir("./mapper_applications/figures_COIL_epsilon=0")
        
# Innitiate different overlaps and filters. Filters need to have .fit() and .transform() functions
overlap_fracs = [0.35, 0.4, 0.45, 0.49]
filters = [
        sklearn.decomposition.PCA(n_components=1)
]
filter_names = [
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
                                                gain=overlap_frac,
                                                epsilon=0)
            print(f"Filter: {filter_name} \nGain: {overlap_frac} \nRips parameter: {delta}\nResolution: {resolution}\nDTM parameter: {k}\nAlpha: {alpha}")            
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
                # Save Mapper figure to disk
                filename = (
                    "./mapper_applications/figures_COIL_epsilon=0/COIL_"
                    + f"{filter_name}_{clusterer_name}_{overlap_frac}_{alpha}.svg"
                )
                fig.write_image(filename)

