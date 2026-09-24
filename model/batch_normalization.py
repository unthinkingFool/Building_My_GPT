import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)
        if training:
            step=x.shape[0]
            n=x.shape[1]
            mean=[]
            var=[]
            for i in range(n):
                mean_i=np.mean((x[:,i]))
                mean.append(mean_i)
                var_i = np.mean((x[:, i] - mean_i) ** 2)
                var.append(var_i)

            mean = np.array(mean)
            var = np.array(var)

            denom=(var+eps)**0.5
            nom=x-mean
            x_hat=nom/denom

            m=momentum
            running_mean=(1-m)*running_mean+m*mean
            running_var=(1-m)*running_var+m*var

        if not training:
            x_hat=(x-running_mean)/((running_var+eps)**0.5)

        y=gamma*x_hat+beta
        
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
        


