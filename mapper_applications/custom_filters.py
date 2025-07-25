import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from gtda.utils._docs import adapt_fit_transform_docs

class Norm(BaseEstimator, TransformerMixin):

    def __init__(self, exponent=2):
        self.exponent = exponent
    
    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X)
        Xt = np.linalg.norm(Xt, axis=1, ord=self.exponent, keepdims=True)
        return Xt

