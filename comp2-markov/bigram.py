import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(69)

class BigramModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    
    def forward(self, idx, targets=None):

        logits = self.embeddings(idx) # (B * T * C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_length):
        # idx is (B * T) tensor of indices in current context
        for _ in range(max_length):
            logits, loss = self(idx) # get predictions
            logits = logits[:, -1, :] # get last prediction (B * C)
            probs = F.softmax(logits, dim=-1) # get probabilities
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=-1) # append to idx
        return idx
            
