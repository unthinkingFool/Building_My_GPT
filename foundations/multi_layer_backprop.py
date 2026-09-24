import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)

        W2 = np.array(W2)
        b2 = np.array(b2)

        y_true = np.array(y_true)
        
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1=np.dot(W1,x)+b1
        a1=np.maximum(0,z1)

        z2=np.dot(W2,a1)+b2
        y_hat=z2

        # Loss: MSE = mean((predictions - y_true)^2)

        loss=np.mean((y_hat-y_true)**2)

        #

        dz2 = 2*(y_hat - y_true)/y_true.size

        dw2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = np.dot(W2.T, dz2)

        dz1 = da1 * (z1 > 0)

        dw1 = np.outer(dz1, x)
        db1 = dz1
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return {"loss": float(np.round(loss,4)),"dW1": np.where(np.abs(np.round(dw1,4)) < 1e-12, 0, np.round(dw1,4)).tolist(),"db1":np.round(db1,4).tolist(),"dW2": np.round(dw2,4).tolist(),"db2":np.round(db2,4).tolist()}
