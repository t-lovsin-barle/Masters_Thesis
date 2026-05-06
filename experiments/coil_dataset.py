import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import gtda.mapper as mpr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(project_root))

from core.custom_cover import ResolutionCover
from core.custom_clusterer import RipsClustering
from core.helper_functions import compute_parameters
from external.automato.automato import Automato

DATA_DIR = project_root / "external" / "COIL_100_Duck"
FIG_DIR = project_root / "figures" / "COIL" / "epsilon=0"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# Download: https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
# Path to the image folder


data = []
# Loop through obj74 images
for file in os.listdir(DATA_DIR):
    
        angle = int(file.stem.split("__")[-1]

        # Load image, convert to grayscale and scale pixel values to [0, 1]
        img = Image.open(file).convert('L')  # 'L' for grayscale
        img_array = np.array(img) / 255.0  # normalize to [0, 1]
        img_flat = img_array.flatten(order='F')
        
        # Compute label
        label = 1 - abs(angle - 180) / 180

        # Add to data
        data.append(np.append(img_flat, label))

# Create DataFrame
df = pd.DataFrame(data)
X, y = df.values[:, :-1], df.values[:, -1]        

overlap_fracs = [0.35, 0.4, 0.45, 0.49]
filters = [
        sklearn.decomposition.PCA(n_components=1)
]
filter_names = [
        "PCA"
]

n_jobs = 1

for overlap_frac in overlap_fracs:
    for filter_func , filter_name in zip(filters, filter_names):
        
        delta, resolution, k = compute_parameters(X=X,
                                            filter_func=filter_func.fit(X).transform(X).flatten(),
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
            filename = FIG_DIR / f"COIL_{filter_name}_{clusterer_name}_{overlap_frac}.svg"
            fig.write_image(str(filename))

