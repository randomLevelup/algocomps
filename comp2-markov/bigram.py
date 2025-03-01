import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ Single self-attention head """
    def __init__(self, device, n_embed, head_size, block_size, bias=False):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,
                                                           block_size,
                                                           device=device)))
        
        # self.to(device)
        
    def forward(self, x):

        # apply self-attention
        B,T,C = x.shape
        K = self.key(x)
        Q = self.query(x)

        wts = Q @ K.transpose(-2, -1) * (C ** -0.5) # soften weights by head_size
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # matmul averaging trick to compute affinities
        wts = wts.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # mask future tokens
        wts = F.softmax(wts, dim=-1)

        # aggregate affinities
        V = self.value(x) # (B, T, head_size)
        out = wts @ V # (B, T, head_size) @ (B, head_size, C) -> (B, T, C)
        return out


class BigramModel(nn.Module):
    """ Simple bigram model """
    def __init__(self, device, block_size, vocab_size, n_embed):
        super().__init__()
        
        self.device = device
        self.block_size = block_size  # store block_size for use in generate method
        self.n_embed = n_embed
        
        # create the embeddings and layers
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(device, n_embed, n_embed, block_size)
        self.output = nn.Linear(n_embed, vocab_size)
        
        # move everything to the specified device
        self.to(device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # get token and position embeddings
        tok_emb = self.token_embeddings(idx)  # (B, T, C)
        pos = torch.arange(T, device=self.device)
        pos_emb = self.position_embeddings(pos)  # (T, C)
        
        # add positional embeddings
        x = tok_emb + pos_emb  # (B, T, C)
        
        # apply self-attention head
        x = self.sa_head(x)  # (B, T, C)

        # shape outputs (scores)
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

