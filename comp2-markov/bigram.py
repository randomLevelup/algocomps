import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramModel(nn.Module):

    def __init__(self, device, block_size, vocab_size, n_embed):
        super().__init__()
        
        self.device = device
        self.block_size = block_size  # store block_size for use in generate method
        
        # create the embeddings and layers
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, n_embed)
        self.output = nn.Linear(n_embed, vocab_size)
        
        # move everything to the specified device
        self.to(device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # get token and position embeddings
        tok_emb = self.token_embeddings(idx)  # (B, T, C)
        pos = torch.arange(T, device=self.device)  # ensure positions are on correct device
        pos_emb = self.position_embeddings(pos)  # (T, C)
        
        # add positional embeddings
        x = tok_emb + pos_emb  # (B, T, C)
        
        # apply linear layer and get logits
        x = self.lm_head(x)  # (B, T, C)
        logits = self.output(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_length):
        # idx is (B, T) tensor of indices in current context
        for _ in range(max_length):

            # truncate context to block_size if it exceeds the limit
            idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
            
            # get model predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # get last token's logits (B, vocab_size)
            
            # sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # append sampled token to the sequence
            idx = torch.cat([idx, next_idx], dim=1)
        
        return idx

