import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        all_sen=positive+negative
        words=set()
        for sen in all_sen:
            for word in sen.split():
                words.add(word)
        words=sorted(words)
        vocab={
            word:idx
            for idx,word in enumerate(words,start=1)
        }
        encoding=[]
        for sen in all_sen:
            en=[vocab[word] for word in sen.split()]
            encoding.append(en)

        tensors = [
                torch.tensor(sentence)
                for sentence in encoding
        ]
   
        dataset = nn.utils.rnn.pad_sequence(
                tensors,
                batch_first=True,
                padding_value=0
        )
        return dataset



