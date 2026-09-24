import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        ans=[]
        
        with torch.no_grad():
            out=x
            for layer in model: 
                out=layer(out)
                if isinstance(layer,nn.ReLU):
                    

                    check=(out==0).all(dim=0).float().mean().item()
                    ans.append(round(check,4))

        return ans

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        for l in dead_fractions:
            if l>0.5 :
                return "use_leaky_relu"
        if dead_fractions[0]>0.3: 
            return "reinitialize"
        
        check =all(dead_fractions[i]<dead_fractions[i+1] for i in range(len(dead_fractions)-1))

        if check and dead_fractions[-1]>0.1 : 
            return "reduce_learning_rate"

        if max(dead_fractions)<0.1 : 
            return "healthy"
        return "healthy"










