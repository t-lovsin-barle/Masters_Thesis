
# Code to recreate results of Mapper applied to diabetes data
import clustbench
import sys
import os
# Add the project root to sys.path so 'automato' can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, '..', '..'))
sys.path.insert(0, project_root)
import gtda.mapper as mpr  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from automato import Automato
from mapper_applications.eccentricity_subclassed import EccentricitySubclassed
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
data_path = r"C:\Users\trist\Downloads\clustering-data-v1-1.1.0\clustering-data-v1-1.1.0"

data_set = clustbench.load_dataset('wut','smile',path=data_path)
data=data_set.data
plt.scatter(data[:,0],data[:,1],marker='.',s=50)


plt.show()

a = Automato()
a.fit(data)
a.plot_diagram()

plt.scatter(data[:,0],data[:,1],marker='.',s=50,c=a.labels_)
plt.show()