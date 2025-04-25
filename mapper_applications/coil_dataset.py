# Code to recreate results of Mapper applied to diabetes data

import sys
import os

# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)


import gtda.mapper as mpr  # type: ignore
from custom_cover import ResolutionCover
from custom_clusterer import AutoRipsClusering
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import math
import sklearn

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed
from scipy.spatial.distance import pdist, squareform, directed_hausdorff, cdist

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

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

# Split features and label
X, y = df.values[:, :-1], df.values[:, -1]

def empiric_mod_of_contin(func, delta, dist_mtrx, epsilon=0.001):
    V = 0
    for i in range(len(dist_mtrx)):
        for j in range(i,len(dist_mtrx)):
            if dist_mtrx[i,j] <= delta:
                V_ = abs(func[i]-func[j])
                if V <= V_:
                    V = V_
    return V + epsilon
def delta_(X, beta=0.001, n_iterations=100):
    delta = 0
    s_n = math.floor(len(X) / (np.log(len(X)) ** (1 + beta)))
    for i in range(n_iterations):
        indices = np.random.choice(len(X), size=s_n, replace=False)
        all_indices = np.arange(len(X))
        c_indices = np.setdiff1d(all_indices, indices)
        dist_matrix = cdist(X[c_indices],X[indices])
        hausdorff_dist_from_data_to_sample = 0
        for j in range(len(c_indices)):
            min_j = dist_matrix[j,0]
            for k in range(len(indices)):
                min_j = min(min_j, dist_matrix[j,k])
        hausdorff_dist_from_data_to_sample = max(hausdorff_dist_from_data_to_sample, min_j)
        delta = delta + hausdorff_dist_from_data_to_sample / n_iterations
    return delta

# calculations for relative gap size and eccentricity
dist_mtrx = squareform(pdist(X, metric = 'euclidean'))

func = sklearn.decomposition.PCA(n_components=1).fit(X).transform(X) # For the calculation of the n_intervals
np.fill_diagonal(dist_mtrx, np.inf)



# Instantiate Mapper parameters

overlap_frac = 0.4  # Specify fractional overlap (gain)
V = empiric_mod_of_contin(
    func=func, 
    delta=delta_(X),
    dist_mtrx=dist_mtrx
    )

print(f"V is \n", V)
resolution = V[0] / overlap_frac

clusterer = AutoRipsClusering()
#clusterer = Automato(random_state=42)
#clusterer = Automato(tomato_params={'graph_type':'radius', 'r':delta}, random_state=42)

cover = ResolutionCover(
    resolution=resolution,
    gain=overlap_frac
)
#print(f"cover: \n", cover.fit_transform(X))
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
