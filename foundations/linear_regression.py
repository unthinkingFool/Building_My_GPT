import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        arr=np.dot(X,weights)
        return np.round(arr,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        # the array is like (n*1) size ----> we need to flatten this to get 1D array
        temp_arr=(model_prediction-ground_truth).flatten()
        ans=np.dot(temp_arr,temp_arr)
        ans=ans/model_prediction.shape[0]

        return np.round(ans,5)
