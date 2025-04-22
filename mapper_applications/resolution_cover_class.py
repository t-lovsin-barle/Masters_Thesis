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

class OneDimResolutionCover(BaseEstimator, TransformerMixin):

    _hyperparameters = {
        'resolution': {'type': float},
        'gain': {'type': float}
    }

    def __init__(self, resolution= None, gain= None):
        self.resolution = resolution
        self.gain = gain
    
    def fit(self, X, y=None): # not finished
        X = check_array(X, ensure_2d=False)
        #validate_params(self.get_params(), self._hyperparameters)
        if self.overlap_frac <= 1e-8:
            warnings.warn("`overlap_frac` is close to zero, "
                        "which might cause numerical issues and errors.",
                        RuntimeWarning)
            
        #if X.ndim == 2:
        #    _check_has_one_column(X)



        return self
    
    def _find_interval_limits(self, X, resolution, gain):
        min_val, max_val = np.min(X), np.max(X)
        centre = (max_val - min_val) / 2

        left_limits, right_limits = self._cover_limits(min_val, max_val, centre, resolution, gain)
        
        return left_limits, right_limits
    
    def _cover_limits(min_val, max_val, centre, resolution, gain):
        range_len = max_val - min_val
        interval_nr = math.ceil(range_len / resolution) + 2 
        step = resolution * (1 - gain)
        
        if interval_nr % 2 == 1:
            




        return left_limits, right_limits
    