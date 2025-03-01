import torch
import pickle
from tqdm import tqdm
from music21 import *
from preprocess import *
from bigram import *

# hyperparameters
batch_size     = 32   # num batches to process in paralell
block_size     = 16    # max context length for predictions
max_iters      = 4000
lr             = 1e-2 # learning rate
wd             = 1e-2 # weight decay
eval_iters     = 200
eval_interval  = 100
key_variations = 4

vocab_size = 129 * 25 # (128 MIDI notes + 1 rest token) * 25 possible durations
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------

torch.manual_seed(69)

# load data
data_dir = '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/data'
pickle_path = 'comp2-markov/data.pkl'

def process_and_backup_data():
    input_sequences = preprocess_data(data_dir, variation_amt=key_variations)
    print("\nLoaded input data:", len(input_sequences))
    flat_input = [item for sublist in input_sequences for item in sublist]
    input_tokens = make_tokens(flat_input)
    with open(pickle_path, 'wb') as f:
        pickle.dump((key_variations, input_tokens), f)
    return input_tokens


if os.path.exists(pickle_path):
    with open(pickle_path, 'rb') as f:
        pkl_numkeys, pkl_tokens = pickle.load(f)
        if pkl_numkeys == key_variations:
            input_tokens = pkl_tokens
        else:
            input_tokens = process_and_backup_data()
else:
    input_tokens = process_and_backup_data()
    

full_data = torch.tensor(input_tokens, dtype=torch.long)

print("\nData loaded. Full data shape:")
print(full_data.shape, full_data.dtype)

split_idx = int(len(input_tokens) * 0.9)
train_data, val_data = full_data[:split_idx], full_data[split_idx:]
print("\nTrain data:", train_data.shape)
print("Val data:", val_data.shape)

print("\nLoading model...")
model = BigramModel(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
print("Good.")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\nTraining...")
progress_bar = tqdm(range(max_iters), desc="Training...")

for iter in range(max_iters):

    # evaluate train & val sets occasionally
    if iter % eval_interval == 0:
        losses = estimate_loss()
        progress_bar.set_postfix({"train": losses['train'].item(), "val": losses['val'].item()})
    
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    progress_bar.update(1)

progress_bar.close()
print("Done. Final loss:", loss.item())

print("\nGenerating...")
gen_tokens = model.generate(
    idx=torch.zeros((1, 1), dtype=torch.long).to(device),
    max_length=100)[0].tolist()
gen_stream = sequence_to_score(tokens_to_sequence(gen_tokens))
gen_stream.write('midi', fp='comp2-markov/generations/gen.mid')
print("saved 'gen.mid'")
