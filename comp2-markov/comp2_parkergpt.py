"""

File: comp2_parkergpt.py

Defines the ParkerGPT neural network model and training functions.

"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm

# default_hyperparameters
def get_default_hyperparams():
    return {
        'batch_size'     : 64,   # number of sequences to train on in parallel
        'block_size'     : 8,    # max context length for predictions
        'num_layers'     : 6,    # number of transformer blocks
        'num_heads'      : 6,    # number of self-attention heads per block
        'max_iters'      : 4000, # number of training iterations
        'lr'             : 5e-4, # learning rate
        'wd'             : 1e-3, # weight decay
        'dropout'        : 0.3,  # dropout rate
        'eval_iters'     : 200,
        'eval_interval'  : 500,
        'num_embeddings' : 384,  # embedding dimension: num_heads * 64
        'key_variations' : 4,
        'vocab_size'     : 129 * 25 # (128 MIDI notes + 1 rest token) * 25 possible durations
    }

class Head(nn.Module):
    """ Single self-attention head """
    def __init__(self, device, n_embed, head_size, block_size, dropout, bias=False):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,
                                                           block_size,
                                                           device=device)))
        self.dropout = nn.Dropout(dropout) # dropout layer
                
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
        wts = self.dropout(wts) # apply dropout to weights

        # aggregate affinities
        V = self.value(x) # (B, T, head_size)
        out = wts @ V # (B, T, head_size) @ (B, head_size, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ Multi-head self-attention """
    def __init__(self, device, n_embed, head_size, block_size, n_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(device, n_embed, head_size, block_size, dropout)
                                    for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embed) # residual connection
        self.dropout = nn.Dropout(dropout) # dropout layer
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ Simple linear layer and a ReLU nonlinearity """
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # feedforward layer
            nn.ReLU(),                       # nonlinearity
            nn.Linear(4 * n_embed, n_embed), # residual connection
            nn.Dropout(dropout)              # dropout layer
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block """
    def __init__(self, device, n_embed, block_size, n_heads, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(device, n_embed, head_size, block_size, n_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed) # layer normalization
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x): # aggregate transformer block with residual connection
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramModel(nn.Module):
    """ Simple bigram model """
    def __init__(self, device, block_size, vocab_size, n_embed, n_layers, n_heads, dropout):
        super().__init__()
        
        self.device = device
        self.block_size = block_size # store block_size used in generation
        
        # create the embeddings and layers
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)

        # multiple transformer blocks: each with several heads of self-attention,
        self.blocks = nn.Sequential(*[
            Block(device, n_embed, block_size, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embed) # final layer normalization
        self.output = nn.Linear(n_embed, vocab_size)
        
        # move everything to the specified device !! important !!
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
        logits = self.output(self.ln_f(x))  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_length):
        # self.eval()

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

def get_batch(which_split, direction, data_splits, hp, device):
    data = None
    if direction == 'forward':
        data = data_splits['train_data_f'] if which_split == 'train' else data_splits['val_data_f']
    else:
        data = data_splits['train_data_b'] if which_split == 'train' else data_splits['val_data_b']

    ix = torch.randint(len(data) - hp['block_size'], (hp['batch_size'],))
    x = torch.stack([data[i:i+hp['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+hp['block_size']+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, direction, data_splits, hp, device):
    out = {}
    model.eval()
    for which_split in ['train', 'val']:
        losses = torch.zeros(int(hp['eval_iters']))
        for k in range(int(hp['eval_iters'])):
            X, Y = get_batch(which_split, direction, data_splits, hp, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[which_split] = losses.mean()
    model.train()
    return out


def train_models(input_tokens, device, hp, model_save_dir):
    full_data_forward = torch.tensor(input_tokens, dtype=torch.long)
    full_data_backward = torch.flip(full_data_forward, [0])
    print(f"Train data loaded. Full data shape:")
    print(full_data_forward.shape, full_data_forward.dtype)

    split_idx = int(len(input_tokens) * 0.9)
    data_splits = {
        'train_data_f': full_data_forward[:split_idx],
        'val_data_f': full_data_forward[split_idx:],
        'train_data_b': full_data_backward[:split_idx],
        'val_data_b': full_data_backward[split_idx:]
    }
    print("\nTrain data:", data_splits['train_data_f'].shape)
    print("Val data:", data_splits['val_data_f'].shape)

    print(f"\nLoading model...")
    block_size     = int(hp['block_size'])
    vocab_size     = int(hp['vocab_size'])
    n_layers       = int(hp['num_layers'])
    n_heads        = int(hp['num_heads'])
    num_embeddings = int(hp['num_embeddings'])
    lr             = float(hp['lr'])
    wd             = float(hp['wd'])
    dropout        = float(hp['dropout'])
    max_iters      = int(hp['max_iters'])

    model_f = BigramModel(device, block_size, vocab_size, num_embeddings, n_layers, n_heads, dropout)
    opt_f = torch.optim.AdamW(model_f.parameters(), lr=lr, weight_decay=wd)
    model_b = BigramModel(device, block_size, vocab_size, num_embeddings, n_layers, n_heads, dropout)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=lr, weight_decay=wd)
    print("Done.")

    for model, opt, direction in [(model_f, opt_f, 'forward'), (model_b, opt_b, 'backward')]:
        progress_bar = tqdm(range(max_iters),
                            desc=f"Training: {direction}",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')

        for iter in range(max_iters):

            # evaluate train & val sets occasionally
            if iter % int(hp['eval_interval']) == 0:
                losses = estimate_loss(model, direction, data_splits, hp, device)
                progress_bar.set_postfix({"train": losses['train'].item(), "val": losses['val'].item()})
            
            # train step
            xb, yb = get_batch('train', direction, data_splits, hp, device)
            logits, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            progress_bar.update(1)

        progress_bar.close()
        print("Done. Final loss:", loss.item())

        model_filename = f"parkergpt_{direction}.pt"

        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_embed': num_embeddings,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'dropout': dropout,
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss.item(),
        }, model_save_dir + model_filename)
        print(f"Model saved to {model_save_dir + model_filename}\n")
