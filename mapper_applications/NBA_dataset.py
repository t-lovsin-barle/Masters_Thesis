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


import gudhi
from gudhi.datasets.remote import fetch_spiral_2d
data = fetch_spiral_2d()
import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1],marker='.',s=1)


plt.show()

from gudhi.clustering.tomato import Tomato
t = Tomato()
t.fit(data)
t.plot_diagram()

plt.scatter(data[:,0],data[:,1],marker='.',s=1,c=t.weights_)
plt.show()

plt.scatter(data[:,0],data[:,1],c=t.labels_, cmap='viridis', marker='.', s=1)
plt.show()

to = Tomato(n_clusters=2)
to.fit(data)
to.plot_diagram()

plt.scatter(data[:,0],data[:,1],c=to.labels_, cmap='viridis', marker='.', s=1)
plt.show()
