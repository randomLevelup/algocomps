"""

comp2.py
by jupiter westbard

"""

import argparse, os, pickle, yaml
from music21 import *

from comp2_markov import *
from comp2_preprocess import *
from comp2_parkergpt import *


def main(args):
    mode = 'training' if args.train_model else 'inference'
    print(f"Loading ParkerGPT in {mode} mode...")

    if mode == 'training':
        if args.train_data_dir is None:
            print("A path to valid training data must be specified.")
            print("Rerun with '--train_data_dir [DIR]'")
            quit()
        
        model_save_dir: str = args.model_save_dir
        if model_save_dir[-1] != "/":
            model_save_dir += "/"

        hp: dict = get_default_hyperparams()
        if args.model_config is None:
            print("Using default hyperparameters")
        else:
            try:
                with open(args.model_config, 'r') as cfg_file:
                    config = yaml.safe_load(cfg_file)
                hp = config['hyperparameters']
            except:
                print(f"Error: could not load {args.model_config} as config")
                quit()

            print(f"Using parameters from {args.model_config}")

        if os.path.exists('./data.pkl'):
            with open('./data.pkl', 'rb') as pkl:
                pkl_numkeys, pkl_tokens = pickle.load(pkl)
                if pkl_numkeys == hp['key_variations']:
                    print("Using preprocessed data checkpoint (from './data.pkl')")
                    input_tokens = pkl_tokens
                else:
                    print("Preprocessing data...")
                    input_tokens = process_and_backup_data(args.train_data_dir, hp['key_variations'])
        else:
            input_tokens = process_and_backup_data(args.train_data_dir, hp['key_variations'])
        
        train_models(input_tokens, hp, model_save_dir)

    else: # inference mode
        # parse midi file
        melody_in: stream.Stream = parse_midi(args.input_path)
        print(f"original melody: {melody_in.quarterLength} quarter notes long")

        # get simple triads (baseline for markov chain)
        print("\nSimple triads:")
        keysig, simple_triads = get_triads(melody_in)

        # generate the markov table ()
        print(f"\nGenerating markov table")
        markov_table = generate_markov_table() # TODO: this is too convoluted

        print(f"\nApplying {args.chord_gen_depth} markov generation cycles...")
        substituted_chords = substitute_chords(simple_triads, markov_table)
        for pos_i in range(args.chord_gen_depth):
            substituted_chords = substitute_chords(substituted_chords, markov_table)


def __main__():
    parser = argparse.ArgumentParser(description='Generate algorithmic music composition')

    # training args
    parser.add_argument('--train_model', action='store_true', default=False,
                        help="Use this to run the program in 'training' mode")
    parser.add_argument('--model_config', type=str, metavar='CFG', default=None,
                        help='[YAML] config for training hyperparameters')
    parser.add_argument('--train_data_dir', type=str, metavar='DIR', default=None,
                        help='[Path] to directory with musicxml files')
    parser.add_argument('--model_save_dir', type=str, metavar='DIR', default='./',
                        help='[Path] to directory where trained models will be saved')

    # inference args
    parser.add_argument('--input_path', type=str, metavar='DIR', default=None,
                        help="[File Path] to input '.mid' melody (for inference mode)")
    parser.add_argument('--chord_gen_depth', type=int, metavar='D',
                        help='[Integer] number of markov generation cycles to run', default=5)

    args = parser.parse_args()
    return main(args)

if __name__ == '__main__':
    __main__()

