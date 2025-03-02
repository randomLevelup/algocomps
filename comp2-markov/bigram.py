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


class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention """
    def __init__(self, device, n_embed, head_size, block_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(device, n_embed, head_size, block_size)
                                    for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed) # residual connection
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ Simple linear layer and a ReLU nonlinearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # feedforward layer
            nn.ReLU(),                       # nonlinearity
            nn.Linear(4 * n_embed, n_embed)  # residual connection
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block """
    def __init__(self, device, n_embed, block_size, n_heads=4):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(device, n_embed, head_size, block_size, n_heads)
        self.ffwd = FeedForward(n_embed)
        
    def forward(self, x): # aggregate transformer block with residual connection
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x


class BigramModel(nn.Module):
    """ Simple bigram model """
    def __init__(self, device, block_size, vocab_size, n_embed):
        super().__init__()
        
        self.device = device
        self.block_size = block_size # store block_size used in generation
        
        # create the embeddings and layers
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)

        # 3 transformer blocks: each with 4 heads of 8-dimensional self-attention,
        # each followed by a feed-forward layer and a nonlinearity (ReLU)
        self.blocks = nn.Sequential(
            Block(device, n_embed, block_size, n_heads=4),
            Block(device, n_embed, block_size, n_heads=4),
            Block(device, n_embed, block_size, n_heads=4),
        )

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
        
        # apply transformer blocks
        x = self.blocks(x)  # (B, T, C)

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

