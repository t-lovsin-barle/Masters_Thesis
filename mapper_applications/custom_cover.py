import warnings
from functools import partial
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from gtda.mapper.utils._cover import _check_has_one_column, \
    _remove_empty_and_duplicate_intervals
from gtda.utils.validation import validate_params

# The classes OneDimResolutonCover and ResolutionCover are based on the giotto-tda
# classes OneDimensionalCover and CubicalCover respectively. The main difference is 
# that in the Resolution covers you controll the interval length instead of the number
# of intervals in order to incorperate the results from Carriere et al. (2018):
# https://jmlr.org/papers/v19/17-291.html


#@adapt_fit_transform_docs
class OneDimResolutionCover(BaseEstimator, TransformerMixin):

    # This is the analogue of OneDimensionalCover from the giotto-tda package. 
    # The original C++ implementation of this cover can be found here:
    # https://gudhi.inria.fr/doc/latest/_g_i_c_8h_source.html

    # The construction works as follows: Starting at min_f we create intervals of
    # length = resolution and overlap = gain * resolution. The last interval is then
    # cut off at max_f.

    # This differs from OneDimentionalCover in a meaningful way. Since 
    # OneDimentionalCover distributes the intervals uniformly you cannot guarantee 
    # that the intervals will have length = resolution. This will only be the case if 
    # max_f - min_f is a multiple of the resolution which is rarely the case.
    # It is also not viable to round up or down since that will also impact the 
    # interval length.

    # There exists a second covering mehtod in OneDimentionalCover which also seems 
    # like it is not possible to controll the length of intervals.

    # Parameters
    # ----------

    # gain: A float in (0,1). According to he aforementioned paper we only have 
    # statistical guarantees if gain is set between 1/3 and 1/2.

    # resolution: A float that is equal to V(delta) / gain, where
    # V(delta) = max{ |f(x)-f(x')| : x,x' data points with d(x,x') <= delta} which 
    # mimics the mode of continuity of the filter function. Delta is defined as 
    # the Hausdorff distance between a random subsample of the data and the data.

    # All definitions are based on equation (8) in Carriere et al. (2018)

    _hyperparameters = {
        'resolution': {'type': float},
        'gain': {'type': float}
    }

    def __init__(self, resolution= None, gain= None):
        self.resolution = resolution
        self.gain = gain
    

    def _fit_mock(self, X):
        
        # OneDimensionalCover has two types of covers. Even thoughOneDimResolutionCover 
        # only has one it was simpler for me to mimic the (working) structure of 
        # OneDimentionalCover instead of trying to reinvent the wheel too much, even if
        # this structure might be redundant.

        # _fit_mock is modeled after _fit_uniform in OneDimentionalCover
    

        self.left_limits_, self.right_limits_ = self._find_interval_limits(X, self.resolution, self.gain)
        return self
    
    def fit(self, X, y=None): 

        # fit is exactly the same as in OneDimentional cover. It computes all cover 
        # interval limits according to 'X' and stores them in left_limits_ and 
        # right_limits_. 

        # This method is here to implement the usual scikit-learn API and hence 
        # work in pipelines.

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

        # When running the code the np.logical_and _remove_empty_and_duplicate_intervals
        # clashed in such a way that only one interval remained. This is a fix suggested
        # by AI.

        Xt = np.column_stack([
            np.logical_and(X[:, 0] > left, X[:, 0] < right)
            for left, right in zip(self.left_limits_, self.right_limits_)
            ])
        return Xt
        
    def transform(self, X, y=None):

        # The same as in OneDimentionalCover apart from removing the case distinctions 
        # present in OneDimentionalCover.

        # Compute a cover of `X` according to the cover of the real line
        # computed in :meth:`fit`, and return it as a two-dimensional boolean
        # array. Each column indicates the location of entries in `X`
        # belonging to a common cover interval.        

        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False)

        if X.ndim == 2:
            _check_has_one_column(X)
        else:
            Xt = Xt[:, None]
        
        Xt = self._transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt
    
    def _fit_transform(self, X):

        # This performs the 'uniform' case in OneDimentionalCover.

        Xt = self._fit_mock(X)._transform(X)
        return Xt

    def fit_transform(self, X, y = None, **fit_params):

        # Exactly the same as in OneDimentionalCover. It fits the data, then
        # transforms it.

        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        if Xt.ndim == 2:
           _check_has_one_column(Xt)
        else:
            Xt = Xt[:, None]

        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
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

        # Here a couple of edge cases were removed from OneDimentionalCover, which 
        # sould not be an issue if the user is careful. Otherwise it performs the 
        # 'uniform' procedure from OneDimentionalCover.

        min_val, max_val = np.min(X), np.max(X)        
        left_limits, right_limits = self._cover_limits(min_val, max_val, resolution, gain)
        left_limits[0], right_limits[-1] = -np.inf, np.inf # Ensures that the entire real line is covered
        
        return left_limits, right_limits
    
    @staticmethod
    def _cover_limits(min_val, max_val, resolution, gain):

        # Here the main difference is located. The code is based on the for loop of 
        # the set_cover_from_function function from the C++ source code. The adaptation
        # is that this outputs two arrays, each containing left limits and right limits
        # of intervals respectively, instead of outputing an array of tuples as in the
        # C++ code.

        # Starting at min_val we move to the right in steps where the left limits of the
        # intervals will be located until either an interval would cross max_val OR 
        # we cannot fit another interval into the range of f without only cutting off
        # the last gain fraction of the interval. Outside of the while loop we then add
        # one last interval covering the rest of the range.

        step = resolution * (1 - gain)
        left = min_val
        counter = 1
        while left + resolution <= max_val and max_val + gain * resolution - (left + resolution) >= resolution:
            left = left +step
            counter = counter + 1
        left = left + step
        counter = counter + 1
        left_limits = np.linspace(min_val, left, num=counter, endpoint=True)
        right_limits = left_limits + resolution
        return left_limits, right_limits
    
#@adapt_fit_transform_docs
class ResolutionCover(BaseEstimator, TransformerMixin):

    # Analogosly to CubicalCover, ResolutionCover fits OneDimResolutionCover to 
    # each column of the input array, according to the same parameters passed 
    # to the constructor. All differences are of the same vein as the differences 
    # in OneDimResolutionCover.

    # Further explanation can be found in the comments in OneDimResolutionCover.

    # Parameters
    # ----------

    # gain: A float in (0,1). According to he aforementioned paper we only have 
    #   statistical guarantees if gain is set between 1/3 and 1/2.

    # resolution: A float that is equal to V(delta) / gain, where
    #   V(delta) = max{ |f(x)-f(x')| : x,x' data points with d(x,x') <= delta} which 
    #   mimics the mode of continuity of the filter function. Delta is defined as 
    #   the Hausdorff distance between a random subsample of the data and the data.

    # All definitions are based on equation (8) in Carriere et al. (2018)

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

        Xt = self._transform(Xt)
        return Xt
    
    def fit_transform(self, X, y=None, **fit_params):

        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        # Reshape filter function values derived from FunctionTransformer
        if Xt.ndim == 1:
            Xt = Xt[:, None]

        # This performs the 'uniform' case from CubicalCover
        Xt = self._fit(Xt)._transform(Xt)
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