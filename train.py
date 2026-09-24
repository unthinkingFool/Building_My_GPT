import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

        loss=None

        for epoch in range(epochs):

            torch.manual_seed(epoch)

            indices=torch.randint(0,len(data)-context_length,(batch_size,))

            context=torch.stack([
                data[start:start+context_length]
                for start in indices
            ])
            target=torch.stack([
                data[start+1:start+1+context_length]
                for start in indices
            ])

            logits=model(context)

            logits=logits.view(-1,logits.size(-1))
            target=target.view(-1)

            loss=F.cross_entropy(logits,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(),4)
