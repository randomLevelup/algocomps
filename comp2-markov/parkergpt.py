from preprocess import *

input_sequences = preprocess_data(
    '/mnt/c/Users/jwest/Desktop/algocomps/comp2-markov/data',
    variation_amt=1
)

flat_input = [item for sublist in input_sequences for item in sublist]
input_tokens = make_tokens(flat_input)
