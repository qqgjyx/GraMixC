"""
Distance&indices 2 D transformer.
"""

# Copyright (c) 2024 Juntang Wang

import numpy as np
from scipy.sparse import csc_matrix

class BDTransformer():
    """
    Class for distance&indices 2 D transformer.
    
    Parameters
    ----------
    k: int, default=15
        Number of neighbors for each sample.
        
    Function
    -------
    transform(X)
        X: (distances, indices)
        Transform distance&indices to D.
        Returns: csc_matrix, shape (n_samples, n_samples)
        
    fit(X)
        X: (distances, indices)
        Fit the transformer.
        Returns: self
        
    See `pyc4h.neighbors.BATransformer` for more details.
    Converting imitating `sklearn.neighbors._graph.KNeighborsMixin`.
        
    .. versionadded:: 0.2
    """
    def __init__(self, k=15):
        self.k = k
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform distance&indices to D.
        
        Inputs
        ------
        X: (distances, indices)
            Distance&indices.
            
        Returns
        -------
        D: csc_matrix, shape (n_samples, n_samples)
            The column stochastic matrix.
        """
        k = self.k
        distances, indices = X
        if distances.shape[1]<k or indices.shape[1]<k:
            raise ValueError(f"Number of neighbors must be greater than or equal to k, got {np.max(distances.shape[1], indices.shape[1])}.")
        if distances.shape[0] != indices.shape[0] or distances.shape[1] != indices.shape[1]:
            raise ValueError("distances and indices must have the same shape.")
        
        distances = distances[:, :k]
        indices = indices[:, :k]
    
        # ---------------------------------------------------------------------
        # From sklearn.neighbors._base.KNeighborsMixin.kneighbors_graph line 1011 to 1029
        # ---------------------------------------------------------------------
        D_data, D_ind = distances, indices
        D_data = np.ravel(D_data)
        
        n_queries = D_ind.shape[0]
        n_nonzero = n_queries * k
        D_indptr = np.arange(0, n_nonzero + 1, k)
        
        D = csc_matrix(                             # adjusted to csc_matrix
            (D_data, D_ind.ravel(), D_indptr),
            shape=(n_queries, n_queries)
        )

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------

        return D