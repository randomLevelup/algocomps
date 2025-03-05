import torch
import pickle
from tqdm import tqdm
from music21 import *
from preprocess import *
from bigram import BigramModel

# hyperparameters
batch_size     = 16   # number of sequences to train on in parallel
block_size     = 8    # max context length for predictions
max_iters      = 3000
lr             = 1e-2 # learning rate
wd             = 1e-3 # weight decay
eval_iters     = 50
eval_interval  = 200
num_embeddings = 32
key_variations = 4

vocab_size = 129 * 25 # (128 MIDI notes + 1 rest token) * 25 possible durations
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------

torch.manual_seed(100)

# load data
data_dir = '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/data'
pickle_path = '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/data.pkl'
generation_dir = '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/generations/'
model_save_dir = '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/trained_models/'

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
    

full_data_forward = torch.tensor(input_tokens, dtype=torch.long)
full_data_backward = torch.flip(full_data_forward, [0])

print("\nData loaded. Full data shape:")
print(full_data_forward.shape, full_data_forward.dtype)

split_idx = int(len(input_tokens) * 0.9)
train_data_f, val_data_f = full_data_forward[:split_idx], full_data_forward[split_idx:]
train_data_b, val_data_b = full_data_backward[:split_idx], full_data_backward[split_idx:]
print("\nTrain data:", train_data_f.shape)
print("Val data:", val_data_f.shape)

print("\nLoading model...")
model_f = BigramModel(device, block_size, vocab_size, num_embeddings)
opt_f = torch.optim.AdamW(model_f.parameters(), lr=lr, weight_decay=wd)
model_b = BigramModel(device, block_size, vocab_size, num_embeddings)
opt_b = torch.optim.AdamW(model_b.parameters(), lr=lr, weight_decay=wd)
print("Good.")

def get_batch(split, direction):
    data = None
    if direction == 'forward':
        data = train_data_f if split == 'train' else val_data_f
    else:
        data = train_data_b if split == 'train' else val_data_b

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, direction):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, direction)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for model, opt, direction in [(model_f, opt_f, 'forward'), (model_b, opt_b, 'backward')]:
    progress_bar = tqdm(range(max_iters),
                        desc=f"Training: {direction}",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')

    for iter in range(max_iters):

        # evaluate train & val sets occasionally
        if iter % eval_interval == 0:
            losses = estimate_loss(model, direction)
            progress_bar.set_postfix({"train": losses['train'].item(), "val": losses['val'].item()})
        
        # train step
        xb, yb = get_batch('train', direction)
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
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss.item(),
    }, model_save_dir + model_filename)
    print(f"\nModel saved to {model_save_dir + model_filename}")

    gen_filename = f"gen_{direction}.mid"

    print("\nGenerating...")
    gen_tokens = model.generate(
        idx=torch.zeros((1, 1), dtype=torch.long).to(device),
        max_length=100)[0].tolist()
    gen_stream = sequence_to_score(tokens_to_sequence(gen_tokens))
    gen_stream.write('midi', fp=generation_dir + gen_filename)
    print("saved", gen_filename)
