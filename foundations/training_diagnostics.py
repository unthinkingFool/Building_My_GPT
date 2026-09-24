import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stat=[]
        with torch.no_grad() :
            out=x


            for layer in model : 
                out=layer(out)

                if isinstance (layer,nn.Linear):
                    mean=out.mean().item()
                    std=out.std().item()

                    dead=(out<=0).all(dim=0).float().mean().item()


                    stat.append({
                        "mean":round(mean,4),
                        "std":round(std,4),
                        "dead_fraction":round(dead,4),
                    })

        return stat

        

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        pred=model(x)
        loss=nn.MSELoss()(pred,y)
        loss.backward()

        stat=[]

        for layer in model:
            if isinstance(layer,nn.Linear):
                grad=layer.weight.grad

                stat.append({
                    "mean":round(grad.mean().item(),4),
                    "std":round(grad.std().item(),4),
                    "norm":round(torch.norm(grad).item(),4),
                })

        return stat
        



    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for s in activation_stats:
            if s["dead_fraction"]>0.5:
                return "dead_neurons"

        for g in gradient_stats:
            if g["norm"]>1000:
                return "exploding_gradients"

        if gradient_stats[-1]["norm"]<1e-5:
            return "vanishing_gradients"

        for a in activation_stats: 
            if a["std"]<0.1:
                return "vanishing_gradients"
            if a["std"]>10.0:
                return "exploding_gradients"

        return "healthy"































