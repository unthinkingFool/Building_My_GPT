import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.query=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.value=nn.Linear(embedding_dim,attention_dim,bias=False)
        self.attention_dim=attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        q=self.query(embedded)
        k=self.key(embedded)
        v=self.value(embedded)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        attention_score= (q @ k.transpose(-2,-1))/torch.sqrt(
            torch.tensor(self.attention_dim, dtype=torch.float32)
        )
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        mask=torch.tril(torch.ones(attention_score.shape[-2:]))
        #    then masked_fill positions where mask == 0 with float('-inf')
        scores=attention_score.masked_fill(mask==0,float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        scores=torch.softmax(scores,dim=2)
        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round(scores@v, decimals=4)
