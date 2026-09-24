import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)

        denom=fan_in+fan_out
        std=(2.0/denom)**0.5

        return (torch.randn(fan_out, fan_in) * std).round(decimals=4).tolist()

        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std=(2.0/fan_in)**0.5

        return (torch.randn(fan_out,fan_in)*std).round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        weights = []
        for i in range(num_layers):
            fan_in  = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == 'xavier':
                std = (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':
                std = (2.0 / fan_in) ** 0.5
            else:
                std = 1.0

            weights.append(torch.randn(fan_out, fan_in) * std)

        
        x = torch.randn(input_dim)


        stds = []
        for W in weights:
            x = torch.relu(W @ x)
            stds.append(round(float(x.std()), 2))

        return stds 