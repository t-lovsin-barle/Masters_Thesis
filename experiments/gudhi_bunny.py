import sys
import os
import gtda.mapper as mpr 
import math
import sklearn
import numpy as np
from scipy.spatial.distance import cdist
import gudhi.datasets.remote
from sklearn.metrics import pairwise_distances

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]  
sys.path.insert(0, str(project_root))

# Custom object import
from external.automato.automato import Automato
from core.helper_functions import average_delta, empiric_mod_of_contin, optimal_resolution
from core.custom_cover import ResolutionCover
from core.custom_clusterer import RipsClustering

FIG_DIR = project_root / "figures" / "Stanford_Bunny"
# Create figures folder if it doesn't exist
FIG_DIR.mkdir(parents=True, exist_ok=True)

X = gudhi.datasets.remote.fetch_bunny(file_path=None, accept_license=False)

z = X[:, 1]

# Normalize z to range [0, 1]
z_min = z.min()
z_max = z.max()
y = (z - z_min) / (z_max - z_min)


overlap_fracs = [0.35, 0.4, 0.45, 0.49]
filters = [
        mpr.Projection(columns=1),
        #mpr.Eccentricity(exponent=np.inf),
        sklearn.decomposition.PCA(n_components=1)
]
filter_names = [
        "ProjectionY",
        #"Eccentricity",
        "PCA"
]
dist_matrix = pairwise_distances(X, metric='euclidean', n_jobs=-1)
for overlap_frac in overlap_fracs:
    print("Current overlap = " + f"{overlap_frac}")
    for filter_func , filter_name in zip(filters, filter_names):
        print("current filter = " + f"{filter_name}")
        delta = 0
        s_n = math.floor(len(X) / (np.log(len(X)) ** (1 + 0.001)) * np.log(10) ** (1 + 0.001))
        print("s_n is " + f"{s_n}")
        for i in range(10):
            indices = np.random.choice(len(X), size=s_n, replace=False)
            sampled_dist_matrix=dist_matrix[:,indices]
            hausdorff_dist_from_data_to_sample = 0
            print("start of iteration " + f"{i}")
            min_j = 0
            for j in range(len(X)):
                min_j = min(sampled_dist_matrix[j,:])

                hausdorff_dist_from_data_to_sample = max(hausdorff_dist_from_data_to_sample, min_j)
            delta = delta + hausdorff_dist_from_data_to_sample / 10
            print("end of iteration " f"{i}")
        print("delta = " f"{delta}")
        V = 0
        filter_func_ = filter_func.fit(X).transform(X)
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                if dist_matrix[i,j] <= delta:
                    V_ = abs(filter_func_[i]-filter_func_[j])
                    if V <= V_:
                        V = V_
                        print("V updated")
        V = float(V + 0.001)
        resolution = V / overlap_frac
        print("resolution = " + f"{resolution}")
        k = int(np.ceil(len(X) * 0.05))
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
            print("current clusterer = " +f"{clusterer_name}")
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

            filename = FIG_DIR / f"Bunndy_{filter_name}_{clusterer_name}_{overlap_frac}.svg"
            fig.write_image(str(filename))
