import warnings
from functools import partial
from itertools import product

import numpy as np
import math
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from gtda.mapper.utils._cover import _check_has_one_column, \
    _remove_empty_and_duplicate_intervals
#from gtda.utils._docs import adapt_fit_transform_docs
#from gtda.utils.intervals import Interval
from gtda.utils.validation import validate_params


#@adapt_fit_transform_docs
class OneDimResolutionCover(BaseEstimator, TransformerMixin):

    _hyperparameters = {
        'resolution': {'type': float},
        'gain': {'type': float}
    }

    def __init__(self, resolution= None, gain= None):
        self.resolution = resolution
        self.gain = gain
    
    # In the original code there are two types of fit. Here there is only one. 
    # However it seems that the structure is somewhat important and thus is retained here.
    def _fit_mock(self, X):
        self.left_limits_, self.right_limits_ = self._find_interval_limits(X, self.resolution, self.gain)
        return self
    
    def fit(self, X, y=None): 

        """" Explanation here"""

        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)
        if self.gain <= 1e-8:
            warnings.warn("`overlap_frac` is close to zero, "
                        "which might cause numerical issues and errors.",
                        RuntimeWarning)
            
        if X.ndim == 2:
            _check_has_one_column(X)

        fitter = self._fit_mock
        return fitter(X)
    
    def _transform(self, X):
        Xt = np.column_stack([
            np.logical_and(X[:, 0] > left, X[:, 0] < right)
            for left, right in zip(self.left_limits_, self.right_limits_)
            ])
        return Xt
        #return np.logical_and(X > self.left_limits_, X < self.right_limits_)
    
    def transform(self, X, y=None):

        """" Explanation here
        """

        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False)

        if X.ndim == 2:
            _check_has_one_column(X)
        else:
            Xt = Xt[:, None]

        # IF statement from onedimensional cover
        #if self.kind == 'boundary':
        #    self._check_limit_attrs()
        
        Xt = self._transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        #print("Interval masks shape:", Xt.shape)
        #print("Non-empty intervals:", Xt.sum(axis=0))
        return Xt
    
    def _fit_transform(self, X):
        Xt = self._fit_mock(X)._transform(X)
        return Xt

    def fit_transform(self, X, y = None, **fit_params):

        """" Explanation here
        """

        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        if Xt.ndim == 2:
           _check_has_one_column(Xt)
        else:
            Xt = Xt[:, None]

        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        #print("Interval masks shape:", Xt.shape)
        #print("Non-empty intervals:", Xt.sum(axis=0))
        return Xt
    
    def get_fitted_intervals(self):
        check_is_fitted(self)
        return list(zip(self.left_limits_, self.right_limits_))
    
    def _check_limit_attrs(self):
        limit_attrs = ['left_limits_', 'right_limits_']
        has_limits = all([hasattr(self, attr) for attr in limit_attrs])
        if not has_limits:
            raise NotFittedError(
                "When the cover is balanced and n_intervals > 1, the left "
                "and right limits of the cover intervals are not "
                "explicitly calculated during 'fit_transform'. Please "
                "call 'fit' explicitly on the same data before using this "
                "method.")
    
    def _find_interval_limits(self, X, resolution, gain):
        min_val, max_val = np.min(X), np.max(X)

        # innitiate boundary case which is changed if kind is maximal
        
        left_limits, right_limits = self._cover_limits(min_val, max_val, resolution, gain)
        left_limits[0], right_limits[-1] = -np.inf, np.inf
        
        return left_limits, right_limits
    
    @staticmethod
    def _cover_limits(min_val, max_val, resolution, gain):
        step = resolution * (1 - gain)
        left = min_val
        counter = 1
        while left + resolution <= max_val and max_val + gain * resolution - (left + resolution) >= resolution:
            left += step
            counter += 1
        left += step
        counter += 1
        left_limits = np.linspace(min_val, left, num=counter, endpoint=True)
        right_limits = left_limits + resolution
        return left_limits, right_limits
    
#@adapt_fit_transform_docs
class ResolutionCover(BaseEstimator, TransformerMixin):

    _hyperparameters = {
        'resolution': {'type': float},
        'gain': {'type': float}
    }

    def __init__(self, resolution= None, gain= None):
        self.resolution = resolution
        self.gain = gain

    def _clone_and_apply_to_column(self, X, coverer, method_name, i):
        # method is either a fit-type or a fit_transform-type method
        try:
            return getattr(clone(coverer), method_name)(X[:, [i]])
        except ValueError as ve:
            if ve.args[0] == f"Only one unique filter value found, cannot " \
                             f"fit {self.n_intervals} > 1 intervals.":
                raise ValueError(
                    f"Only one unique filter value found along feature "
                    f"dimension {i}, cannot fit {self.n_intervals} > 1 "
                    f"intervals there.")
            else:
                raise ve

    def _fit(self, X):
        coverer = OneDimResolutionCover(
                                        resolution=self.resolution,
                                        gain=self.gain
                                        )
        fitter = '_fit_mock'
        self._coverers = [partial(self._clone_and_apply_to_column, X, coverer, fitter)(i)
                          for i in range(X.shape[1])
                          ]
        self.n_features_fit = X.shape[1]
        return self

    def fit(self, X, y=None): 

        """" Explanation here
        """

        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)
       
        if X.ndim == 1:
            X = X[:, None]

        return self._fit(X)
    
    def _transform(self, X):
        # Calculate 1D cover for each column
        covers = [coverer._transform(X[:, [i]])
                  for i, coverer in enumerate(self._coverers)]

        Xt = self._combine_one_dim_covers(covers)
        return Xt
    
    def transform(self, X, y=None):
        """Compute a cover of `X` according to the cover of Euclidean space
        computed in :meth:`fit`, and return it as a two-dimensional boolean
        array whose each column indicates the location of entries in `X`
        belonging to a common cover interval.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to n_intervals *
            n_features` as empty or duplicated cover sets are removed.

        """
        check_is_fitted(self, '_coverers')
        Xt = check_array(X, ensure_2d=False)

        # Reshape filter function values derived from FunctionTransformer
        if Xt.ndim == 1:
            Xt = Xt[:, None]

        n_features_fit = self._n_features_fit
        n_features = Xt.shape[1]
        if n_features != n_features_fit:
            raise DataDimensionalityWarning(
                f"Different number of columns between `fit` ({n_features_fit})"
                f" and `transform` ({n_features}).")

        # IF statement from original code
        #if self.kind == 'balanced':
            # Test on the first coverer whether the left_limits_ and
            # right_limits_ attributes are present
        #    self._coverers[0]._check_limit_attrs()

        Xt = self._transform(Xt)
        return Xt
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to the data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to `n_intervals *
            n_features` as empty or duplicated cover sets are removed.

        """
        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        # Reshape filter function values derived from FunctionTransformer
        if Xt.ndim == 1:
            Xt = Xt[:, None]

        #if self.kind == 'uniform':
        Xt = self._fit(Xt)._transform(Xt)
        return Xt

        # Calculate 1D cover for each column
        coverer = OneDimResolutionCover(kind=self.kind,
                                      resolution=self.resolution,
                                      gain=self.gain)
        coverers = [clone(coverer) for _ in range(Xt.shape[1])]
        fit_transformer = '_fit_transform'
        covers = [
            partial(self._clone_and_apply_to_column,
                    Xt, coverer, fit_transformer)(i)
            for i, coverer in enumerate(coverers)
            ]
        # Only store attributes if above succeeds
        self._coverers = coverers
        self._n_features_fit = Xt.shape[1]
        Xt = self._combine_one_dim_covers(covers)
        return Xt
    
    @staticmethod
    def _combine_one_dim_covers(covers):
        # Stack intervals for each cover
        intervals = (
            [cover[:, i] for i in range(cover.shape[1])] for cover in covers
            )

        # Calculate masks for pullback cover
        Xt = np.array([np.logical_and.reduce(t)
                       for t in product(*intervals)]).T

        Xt = _remove_empty_and_duplicate_intervals(Xt)
        #print("Interval masks shape:", Xt.shape)
        #print("Non-empty intervals:", Xt.sum(axis=0))
        return Xt