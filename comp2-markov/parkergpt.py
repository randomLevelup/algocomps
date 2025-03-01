import torch
from preprocess import *

print("Preprocessing data...")
input_sequences = preprocess_data(
    '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/data',
    variation_amt=1
)
print("Done")

flat_input = [item for sublist in input_sequences for item in sublist]

print("\nGenerating tokens...")
full_data = torch.tensor(make_tokens(flat_input), dtype=torch.long)
print(full_data.shape, full_data.dtype)
print(full_data[:100])

split_idx = int(0.9 * len(full_data))
train_data, val_data = full_data[:split_idx], full_data[split_idx:]
print(train_data.shape, val_data.shape)