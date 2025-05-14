"""
Transform input data into a bipartite attributed graph (A matrix).

Pipeline:
1. (Optional) kNN: Input data -> distances & indices
2. BDTransformer: distances & indices -> D matrix  
3. SGtSNELambdaEq: D matrix -> A matrix

Adapted from C4H work.
"""

# Authors: Juntang Wang <t@qqgjyx.com>
# License: 

import warnings
from typing import Tuple, Union
import numpy as np
from matplotlib import pyplot as plt

from pysgtsnepi.utils import sgtsne_lambda_equalization

from .helper import assert_A_requirements
from ._bdtransformer import BDTransformer

class BATransformer:
    """Transform input data into a bipartite attributed graph.
    
    Parameters
    ----------
    k : int, default=15
        Number of nearest neighbors to use
    lambda_ : float, default=1
        Lambda parameter for t-SNE perplexity adjustment
    
    Attributes
    ----------
    D_ : ndarray of shape (n_samples, n_samples)
        Distance matrix after transformation
    P_ : ndarray of shape (n_samples, n_samples) 
        Probability matrix after lambda equalization
    A_ : ndarray of shape (n_samples, n_samples)
        Final bipartite attributed graph matrix
        
    Examples
    --------
    >>> from neighbours import BATransformer
    >>> transformer = BATransformer(k=15, lambda_=1)
    >>> A = transformer.fit_transform(distances_and_indices)
    """
    def __init__(self, k: int = 15, lambda_: float = 1) -> None:
        self.k = k
        self.lambda_ = lambda_
        
        # Initialize attributes that will be set during transform
        self.D_ = None
        self.P_ = None 
        self.A_ = None

    def fit(self, X, y=None):
        """No-op for compatibility with sklearn pipeline."""
        return self

    def transform(self, B: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Transform distance/indices tuple into bipartite attributed graph.
        
        Parameters
        ----------
        B : tuple of (ndarray, ndarray)
            Tuple containing (distances, indices) arrays
            
        Returns
        -------
        A_ : ndarray of shape (n_samples, n_samples)
            Bipartite attributed graph matrix
            
        Raises
        ------
        AssertionError
            If resulting graph matrix doesn't meet requirements
        """
        self.D_ = BDTransformer(k=self.k).transform(B)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.P_ = sgtsne_lambda_equalization(self.D_, self.lambda_)
        self.A_ = self.P_ / self.P_.sum(axis=0)
        
        try:
            assert_A_requirements(self.A_, self.k)
        except AssertionError as e:
            print(f"Graph validation failed: {e}")
            
        return self.A_
    
    def visualize(self, markersize: float = 0.01, sort: bool = True, sort_index: Union[np.ndarray, None] = None) -> None:
        """Visualize the distance and graph matrices.
        
        Parameters
        ----------
        markersize : float, default=0.01
            Size of markers in spy plots
        sort : bool, default=True
            Whether to apply spectral reordering before visualization
        sort_index : Union[np.ndarray, None], default=None
            Index to apply spectral reordering before visualization
            
        Raises
        ------
        ValueError
            If transform() hasn't been called yet
        """
        if self.D_ is None:
            raise ValueError("D matrix not computed. Call transform() first.")
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
        D_viz = self.D_.copy()
        A_viz = self.A_.copy()
        
        # spectral reordering before visualizing
        if sort:
            if sort_index is not None:
                # Handle sparse matrices properly
                D_viz = D_viz.tocsr()[sort_index, :].tocsc()[:, sort_index]
                A_viz = A_viz.tocsr()[sort_index, :].tocsc()[:, sort_index]
            else:
                raise NotImplementedError("Spectral reordering not implemented yet.")
                import mheatmap as mhm
                D_viz, _ = mhm.graph.spectral_permute(D_viz, None)
                A_viz, _ = mhm.graph.spectral_permute(A_viz, None)
            
        ax1.spy(D_viz, markersize=markersize)
        ax1.set_title('Distance Matrix')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        ax2.spy(A_viz, markersize=markersize)
        ax2.set_title('Graph Matrix') 
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.show()
