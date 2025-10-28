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
import math
from dataset_utils import plotting
from scipy.spatial.distance import cdist

from automato import Automato
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from helper_functions import average_delta, empiric_mod_of_contin
from sklearn.datasets import make_circles

import numpy as np
import pandas as pd
import gudhi.datasets.remote

X = gudhi.datasets.remote.fetch_bunny(file_path=None, accept_license=False)

z = X[:, 1]

# Normalize z to range [0, 1]
z_min = z.min()
z_max = z.max()
y = (z - z_min) / (z_max - z_min)



#y=sklearn.decomposition.PCA(n_components=1).fit(X).transform(X).flatten()
data = np.hstack((X, y[:, np.newaxis]))

import plotly.express as px
import pandas as pd

df = pd.DataFrame(data, columns=["x", "y", "z", "val"])

fig = px.scatter_3d(df, x="x", y="y", z="z", color="val",
                    color_continuous_scale='viridis', size_max=1)
fig.show()
