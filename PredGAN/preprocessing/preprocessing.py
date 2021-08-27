"""

Preprocessing before creating training dataset.

"""

import numpy as np

__author__ = "Tianyi Zhao"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Tianyi Zhao"
__email__ = "tianyi.zhao20@imperial.ac.uk"
__status__ = "Development"


def calculate_deriv(X_train, step):
    """
    Calculate the 1st order time derivative by (a_n+1 - a_n)/step.
    """
    X_deriv = [[] for i in range(len(X_train)-1)]
    for i in range(len(X_deriv)):
        for j in range(5):
            X_deriv[i].append((X_train[i+1, j] - X_train[i, j]) / step)
    
    return np.array(X_deriv)


def concat_timesteps(X_train, ntimes, step):
    X_train_concat = []
    for i in range(len(X_train)-ntimes*step):
        X_current = X_train[i:i+ntimes*step:step]  # [start_idx:end_idx:step]
        X_train_concat.append(X_current)
    
    return np.array(X_train_concat)