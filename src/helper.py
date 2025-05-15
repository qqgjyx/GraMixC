"""
Helper functions for the project.

Drafted by: Juntang Wang @ Apr 16, 2025

Copyright (c) 2025, Reserved
"""

from typing import Optional
import warnings
import numpy as np
import scipy
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

import mheatmap as mhm

from .neighbours._batransformer import BATransformer

class BaseHelper:
    def __init__(self, output_dir: str, dataset_name: str, ) -> None:
        self.dataset_name = dataset_name
        self.output_dir = output_dir

class CustomHelper(BaseHelper):
    def __init__(self, output_dir: str, dataset_name: str, ) -> None:
        super().__init__(output_dir, dataset_name)

    def process_and_save_graph(
        self,
        X: np.ndarray,
        k: int,
        dataset_type: str,
        lambda_: float = 1,
        markersize: float = 0.01, 
        y: Optional[np.ndarray] = None,   
    ) -> None:
        print(f"\nProcessing {dataset_type} data with k = {k}")
        
        # Compute nearest neighbors
        knn = NearestNeighbors(n_neighbors=k + 1, radius=float('inf'), n_jobs=-1)
        B = knn.fit(X).kneighbors()
        
        # Transform to graph matrices
        bat = BATransformer(k=k, lambda_=lambda_)
        A = bat.transform(B)
        D, P = bat.D_, bat.P_
        
        # Visualize
        if y is not None:
            # sort using the y values
            sort_index = np.argsort(y).numpy()
            bat.visualize(sort_index=sort_index, markersize=markersize)
        else:
            bat.visualize(markersize=markersize)
        
        # Save matrices
        A_file = f'{self.output_dir}/{self.dataset_name}_{dataset_type}_A_{k}.mat'
        scipy.io.savemat(A_file, {'A': A, 'P': P, 'D': D})
        print(f"Matrices saved to {A_file}")
        print(" ✓ Done!")
        print("-"*100)
 

    def process_bluered(
        self, 
        y_classify, 
        k, 
        dataset_type,
        remove_tick_labels: bool = True,
    ) -> None:
        print(f"\nProcessing {dataset_type} data BR results")

        bluered_file = f'{self.output_dir}/{self.dataset_name}_{dataset_type}_bluered_{k}.mat'
        cid_f_0 = scipy.io.loadmat(bluered_file)['cid_f'][0, 0]
        true_labels = y_classify.reshape(-1) + 1 #(0-based to 1-based)

        # Find best configuration by ARI score
        aris = [adjusted_rand_score(true_labels, cid_f_0[:, i]) 
                for i in range(cid_f_0.shape[1])]
        best_index = np.argmax(aris)
        

        print("\n"+"="*50+"\n"+" "*15+"Adjusted Rand Index Scores"+"\n"+"="*50)
        for i, ari in enumerate(aris):
            prefix = "→ " if i == best_index else "  "
            print(f"{prefix}Configuration {i+1:2d}: {ari:.4f}")
        print("-"*50)
        print(f"Best Config: {best_index + 1} (ARI = {aris[best_index]:.4f})")
        print("="*50 + "\n")

        # Process confusion matrix
        pred_labels = cid_f_0[:, best_index]
        _, conf_mat, labels = mhm.amc_postprocess(pred_labels, true_labels)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            rms_C, rms_labels, *_ = mhm.rms_permute(conf_mat, labels)
        rms_labels = np.array(rms_labels - 1, dtype=object)

        # Visualization
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        sns.heatmap(
            rms_C, ax=ax1, xticklabels=rms_labels, yticklabels=rms_labels, cmap='YlGnBu',annot=False, fmt='d',
        )
        mhm.mosaic_heatmap(
            rms_C, ax=ax2, xticklabels=rms_labels, yticklabels=rms_labels, cmap="YlGnBu"
        )
        ax1.set(xlabel='Predicted Label', ylabel='True Label', title='Standard')
        ax2.set(xlabel='Predicted Label', ylabel='True Label', title='RMS Permuted')
        for ax in (ax1, ax2):
            ax.xaxis.set_ticks_position('top')
            if remove_tick_labels:
                ax.set_xticklabels([]) # remove tick labels
                ax.set_yticklabels([]) # remove tick labels
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return cid_f_0, best_index
    
