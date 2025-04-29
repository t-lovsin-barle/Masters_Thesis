from inspect import signature

import numpy as np
import math
import gudhi
from scipy.spatial.distance import cdist
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils import check_array

from gtda.utils.validation import validate_params

# Since the goal is to reproduce the results from Carriere et al. (2018) we need to
# cluster the same way they did. First they build a Rips Graph on the data points and
# then take the connected components as clusters when looking at the preimages. What 
# RipsClustering does is something similar, which should yield the same result. Given a 
# pre-image the RipsClustering then constructs a Rips graph on top of it instead of first
# creating the Rips graph and then cutting it. This should not be neaningfully different
# since the same threshold is used in both cases.

# Carriere et al. (2018): https://jmlr.org/papers/v19/17-291.html

class RipsClustering(ClusterMixin, BaseEstimator):

    # This clusterer is modeled after the FirstSimpleGap clusterer from the giotto-tda
    # library.

    # This clusterer roughly works as follows. First it creates a Rips complex using the 
    # Gudhi library, limiting it to dimention 1. From the Rips complex it creates a graph
    # like structure, such that DFS can be used to find spaning trees, labeling the vertices
    # that belong to the along the way. It then outputs a list of lables, each corresponding
    # to a data point in 'X' to be used in the mapper pipeline.

    # Parameters
    # -----------

    # max_edge_length: The threshold for the Rips Complex. It is computed by by taking 100 random
    #   samples of the data and then calculating the Hausdorff distance from the data to the samples
    #   and then averageing it. The sample size is given in Proposiiton 13 in Carriere et al. (2018).

    _hyperparameters = {'max_edge_length': {'type': float}}

    def __init__(self, max_edge_length=None):
        self.max_edge_length = max_edge_length

    def fit(self, X, y=None):
        
        # Routine taken from the FirstSimpleGap clusterer from giotto-tda.
        X = check_array(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['memory'])

        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            self.n_clusters_ = 1
            return self
        
        # Create the Rips Complex with the bootsraped threshold.
        self.RipsComplex = gudhi.RipsComplex(
                                            points=X,
                                            max_edge_length=self.max_edge_length
                                            ).create_simplex_tree(max_dimension = 1)
        
        # Innitiate graph-like structure.
        self.adjacency = defaultdict(set)
        self.vertices = set()

        # Fill the graph like structure.
        for simplex in self.RipsComplex.get_simplices():
            smplx = simplex[0]
            if len(smplx) == 1:
                self.vertices.add(smplx[0])
            elif len(smplx) == 2:
                u, v = smplx
                self.adjacency[u].add(v)
                self.adjacency[v].add(u)

        labels = {}
        current_label = 0

        # Label the connected components using DFS.
        for v in self.vertices:
            if v not in labels:
                stack = [v]
                while stack:
                    node = stack.pop()
                    if node not in labels:
                        labels[node] = current_label
                        stack.extend(self.adjacency[node])
                current_label += 1
        
        # Convert the list into an array.
        label_list = [-1] * len(self.vertices)
        for v in self.vertices:
            label_list[v] = labels[v]
        
        # Taken from FirstSimpleGap
        self.labels_= label_list
        self.n_clusters_ = len(set(label_list))
        return self

class AutoRipsClustering(ClusterMixin, BaseEstimator):
    
    # Identical to RipsClustering apart from the adition of the bootstrap to compute the delta.

    # Parameters
    # ----------

    # beta: Sample constant used in Proposition 13 in Carriere et al. (2018). By default set to
    #   0.001 as stated in section 5 of the paper.

    # n_iterations: Number of random samples to be taken. By default set to 100 as stated in section
    #   5 of Carriere et al. (2018).

    _hyperparameters = {'beta': {'type': float},
                        'n_iterations':{'type': int}
                        }

    def __init__(self, beta=0.001, n_iterations=100):
        self.beta = beta
        self.n_iterations = n_iterations
    

    def fit(self, X, y=None):

        # Routine taken from the FirstSimpleGap clusterer from giotto-tda.
        X = check_array(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['memory'])

        if X.shape[0] == 1:
            self.labels_ = np.array([0])
            self.n_clusters_ = 1
            return self
        
        # Computation of the threshold for the Rips Complex. The bootstrap is adapted from the
        # set_graph_from_automatic_rips function from the C++ Gudhi source code found here:
        # https://gudhi.inria.fr/doc/latest/_g_i_c_8h_source.html

        self.delta = 0
        s_n = math.floor(len(X) / (np.log(len(X)) ** (1 + self.beta)))
        for i in range(self.n_iterations):
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
            self.delta = self.delta + hausdorff_dist_from_data_to_sample / self.n_iterations
        
        # Create the Rips Complex with the bootsraped threshold.
        self.RipsComplex = gudhi.RipsComplex(
                                            points=X,
                                            max_edge_length=self.delta
                                            ).create_simplex_tree(max_dimension = 1)
        
        # Innitiate graph-like structure.
        self.adjacency = defaultdict(set)
        self.vertices = set()

        # Fill the graph like structure.
        for simplex in self.RipsComplex.get_simplices():
            smplx = simplex[0]
            if len(smplx) == 1:
                self.vertices.add(smplx[0])
            elif len(smplx) == 2:
                u, v = smplx
                self.adjacency[u].add(v)
                self.adjacency[v].add(u)

        labels = {}
        current_label = 0

        # Label the connected components using DFS.
        for v in self.vertices:
            if v not in labels:
                stack = [v]
                while stack:
                    node = stack.pop()
                    if node not in labels:
                        labels[node] = current_label
                        stack.extend(self.adjacency[node])
                current_label += 1
        
        # Convert the list into an array.
        label_list = [-1] * len(self.vertices)
        for v in self.vertices:
            label_list[v] = labels[v]

        # Taken from FirstSimpleGap
        self.labels_= label_list
        self.n_clusters_ = len(set(label_list))
        return self

# The following functions along wiht the class ParallelClustering were left as is.

def _sample_weight_computer(rel_indices, sample_weight):
    return {"sample_weight": sample_weight[rel_indices]}


def _empty_dict(*args):
    return {}


def _indices_computer_precomputed(rel_indices):
    return np.ix_(rel_indices, rel_indices)


def _indices_computer_not_precomputed(rel_indices):
    return rel_indices


class ParallelClustering(BaseEstimator):
    """Employ joblib parallelism to cluster different portions of a dataset.

    An arbitrary clustering class which stores a ``labels_`` attribute in
    ``fit`` can be passed to the constructor. Examples are most classes in
    ``sklearn.cluster``. The input of :meth:`fit` is of the form ``[X_tot,
    masks]`` where ``X_tot`` is the full dataset, and ``masks`` is a 2D boolean
    array, each column of which indicates the location of a portion of
    ``X_tot`` to cluster separately. Parallelism is achieved over the columns
    of ``masks``.

    Parameters
    ----------
    clusterer : object
        Clustering object derived from :class:`sklearn.base.ClusterMixin`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    parallel_backend_prefer : ``"processes"`` | ``"threads"`` | ``None``, \
        optional, default: ``None``
        Soft hint for the selection of the default joblib backend. The default
        process-based backend is 'loky' and the default thread-based backend is
        'threading'. See [1]_.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
       For each point in the dataset passed to :meth:`fit`, a tuple of pairs
       of the form ``(i, partial_label)`` where ``i`` is the index of a boolean
       mask which selects that point and ``partial_label`` is the cluster label
       assigned to the point when clustering the subset of the data selected by
       mask ``i``.

    References
    ----------
    .. [1] "Thread-based parallelism vs process-based parallelism", in
       `joblib documentation
       <https://joblib.readthedocs.io/en/latest/parallel.html>`_.

    """

    def __init__(self, clusterer, n_jobs=None, parallel_backend_prefer=None):
        self.clusterer = clusterer
        self.n_jobs = n_jobs
        self.parallel_backend_prefer = parallel_backend_prefer


    def _validate_clusterer(self):
        """Set :attr:`clusterer_` depending on the value of `clusterer`.

        Also verify whether calculations are to be based on precomputed
        metric/affinity information or not.

        """
        if not isinstance(self.clusterer, ClusterMixin):
            raise TypeError("`clusterer` must be an instance of "
                            "sklearn.base.ClusterMixin.")
        params = [param for param in ['metric', 'affinity']
                  if param in signature(self.clusterer.__init__).parameters]
        precomputed = [(getattr(self.clusterer, param) == 'precomputed')
                       for param in params]
        if not precomputed:
            self._precomputed = False
        elif len(precomputed) == 1:
            self._precomputed = precomputed[0]
        else:
            raise NotImplementedError("Behaviour when metric and affinity "
                                      "are both set to 'precomputed' not yet "
                                      "implemented by ParallelClustering.")

    def fit(self, X, y=None, sample_weight=None):
        """Fit the clusterer on each portion of the data.

        :attr:`clusterers_` and :attr:`clusters_` are computed and stored.

        Parameters
        ----------
        X : list-like of form ``[X_tot, masks]``
            Input data as a list of length 2. ``X_tot`` is an ndarray of shape
            (n_samples, n_features) or (n_samples, n_samples) specifying the
            full data. ``masks`` is a boolean ndarray of shape
            (n_samples, n_portions) whose columns are boolean masks
            on ``X_tot``, specifying the portions of ``X_tot`` to be
            independently clustered.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        sample_weight : array-like or None, optional, default: ``None``
            The weights for each observation in the full data. If ``None``,
            all observations are assigned equal weight. Otherwise, it has
            shape (n_samples,).

        Returns
        -------
        self : object

        """
        X_tot, masks = X
        check_array(X_tot, ensure_2d=True)
        check_array(masks, ensure_2d=True)
        if not np.issubdtype(masks.dtype, bool):
            raise TypeError("`masks` must be a boolean array.")
        if len(X_tot) != len(masks):
            raise ValueError("`X_tot` and `masks` must have the same number "
                             "of rows.")
        self._validate_clusterer()

        fit_params = signature(self.clusterer.fit).parameters
        if sample_weight is not None and "sample_weight" in fit_params:
            self._sample_weight_computer = _sample_weight_computer
        else:
            self._sample_weight_computer = _empty_dict

        if self._precomputed:
            self._indices_computer = _indices_computer_precomputed
        else:
            self._indices_computer = _indices_computer_not_precomputed

        # This seems necessary to avoid large overheads when running fit a
        # second time. Probably due to refcounts. NOTE: Only works if done
        # before assigning labels_single. TODO: Investigate
        self.labels_ = None

        labels_single = Parallel(n_jobs=self.n_jobs,
                                 prefer=self.parallel_backend_prefer)(
            delayed(self._labels_single)(
                X_tot[self._indices_computer(rel_indices)],
                rel_indices,
                sample_weight
                )
            for rel_indices in map(np.flatnonzero, masks.T)
            )

        self.labels_ = np.empty(len(X_tot), dtype=object)
        self.labels_[:] = [tuple([])] * len(X_tot)
        for i, (rel_indices, partial_labels) in enumerate(labels_single):
            n_labels = len(partial_labels)
            labels_i = np.empty(n_labels, dtype=object)
            labels_i[:] = [((i, partial_label),)
                           for partial_label in partial_labels]
            self.labels_[rel_indices] += labels_i

        return self


    def _labels_single(self, X, rel_indices, sample_weight):
        cloned_clusterer = clone(self.clusterer)
        kwargs = self._sample_weight_computer(rel_indices, sample_weight)

        return rel_indices, cloned_clusterer.fit(X, **kwargs).labels_

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit to the data, and return the found clusters.

        Parameters
        ----------
        X : list-like of form ``[X_tot, masks]``
            Input data as a list of length 2. ``X_tot`` is an ndarray of shape
            (n_samples, n_features) or (n_samples, n_samples) specifying the
            full data. ``masks`` is a boolean ndarray of shape
            (n_samples, n_portions) whose columns are boolean masks
            on ``X_tot``, specifying the portions of ``X_tot`` to be
            independently clustered.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        sample_weight : array-like or None, optional, default: ``None``
            The weights for each observation in the full data. If ``None``,
            all observations are assigned equal weight. Otherwise, it has
            shape (n_samples,).

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            See :attr:`labels_`.

        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_


    def transform(self, X, y=None):
        """Not implemented.

        Only present so that the class is a valid step in a scikit-learn
        pipeline.

        Parameters
        ----------
        X : Ignored
            Ignored.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        """
        raise NotImplementedError(
            "Transforming new data with a fitted ParallelClustering object "
            "not yet implemented, use fit_transform instead."
            )


    def fit_transform(self, X, y=None, **fit_params):
        """Alias for :meth:`fit_predict`.

        Allows for this class to be used as an intermediate step in a
        scikit-learn pipeline.

        Parameters
        ----------
        X : list-like of form ``[X_tot, masks]``
            Input data as a list of length 2. ``X_tot`` is an ndarray of shape
            (n_samples, n_features) or (n_samples, n_samples) specifying the
            full data. ``masks`` is a boolean ndarray of shape
            (n_samples, n_portions) whose columns are boolean masks
            on ``X_tot``, specifying the portions of ``X_tot`` to be
            independently clustered.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples,)
            See :attr:`labels_`.

        """
        Xt = self.fit_predict(X, y, **fit_params)
        return Xt