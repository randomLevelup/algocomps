"""

comp2.py
by jupiter westbard

"""

import argparse, os, pickle, yaml, random
import torch
from music21 import *

from comp2_markov import *
from comp2_preprocess import *
from comp2_parkergpt import *

def splice_melody(model_f, model_b, device,
                  melody: stream.Stream,
                  past_context=6, # in number of notes or rests
                  future_context=8, # in timesteps: 24ths of a quarter note
                  max_tokens=20,
                  segment_length=6,
                  crossover_threshold=24) -> stream.Stream:
    
    len_split = segment_length * 24
    out = []
    out_fwd = []
    out_back = []
    pos = 0  # position in melody (in 24ths of a quarter note)
    pos_i = 0 # left boundary for current segment
    pos_j = len_split # right boundary
    idx = 0  # idx for forward model
    take = 'melody'
    
    while pos < melody.duration.quarterLength * 24:
        if take == 'melody':
            if pos_i > len_split:
                # print("end melody segment: len_split")
                # use the melody segments surrounding the slice to generate context
                # generate separate contexts for each models
                context_forward = make_tokens(score_to_sequence(out[-past_context:]))

                # use element index to get future context for backward model
                context_backward = make_tokens(score_to_sequence(
                    melody.flatten().notesAndRests.getElementsByOffset(
                        ((pos + len_split) / 24) + 1,
                        ((pos + len_split) / 24) + future_context
                    )[::-1] # reverse the context
                ))
                # if backward context is empty, pad with rests
                if len(context_backward) == 0:
                    final_rests = []
                    for i in range(future_context):
                        final_rests.append(note.Rest(quarterLength=1))
                    context_backward = make_tokens(score_to_sequence(final_rests))

                gen_tokens_fwd = sequence_to_score(tokens_to_sequence(model_f.generate(
                    idx=torch.tensor(context_forward, dtype=torch.long).unsqueeze(0).to(device),
                    max_length=max_tokens
                )[0].tolist())).flatten().notesAndRests

                gen_tokens_back = sequence_to_score(tokens_to_sequence(model_b.generate(
                    idx=torch.tensor(context_backward, dtype=torch.long).unsqueeze(0).to(device),
                    max_length=max_tokens
                )[0].tolist())[::-1]).flatten().notesAndRests

                pos_i = pos
                pos_j = pos + len_split
                out_fwd = []
                out_back = []
                idx = 0
                take = 'model'
                continue

            # take next element from melody
            elem_fwd = melody.flatten().getElementAtOrBefore(pos / 24)
            if elem_fwd is None:
                break
            out.append(elem_fwd)
            pos += int(elem_fwd.quarterLength * 24)
            pos_i += int(elem_fwd.quarterLength * 24)
            
        else:
            if pos_j - pos_i < crossover_threshold:
                middlerest_dur = max(0, pos_j - pos_i) / 24
                for i in range(len(out_back)):
                    out_back[i].offset -= middlerest_dur

                out.extend(out_fwd)
                out.append(note.Rest(quarterLength=middlerest_dur))
                out.extend(out_back)
                pos_i = 0
                pos += len_split
                take = 'melody'
                continue
                
            # Get next elem by index
            elem_fwd = gen_tokens_fwd[idx]
            
            # Deepcopy notes to prevent ID conflicts
            if isinstance(elem_fwd, note.Note):
                elem_fwd = note.Note(pitch=elem_fwd.pitch, quarterLength=elem_fwd.quarterLength)
            elif isinstance(elem_fwd, note.Rest):
                elem_fwd = note.Rest(quarterLength=elem_fwd.quarterLength)
            
            elem_fwd.offset = pos_i / 24 # Set offset
            pos_i += int(elem_fwd.quarterLength * 24) # Then increment position
            out_fwd.append(elem_fwd) # Then append

            elem_back = gen_tokens_back[idx]
            if isinstance(elem_back, note.Note):
                elem_back = note.Note(pitch=elem_back.pitch, quarterLength=elem_back.quarterLength)
            elif isinstance(elem_back, note.Rest):
                elem_back = note.Rest(quarterLength=elem_back.quarterLength)
            
            pos_j -= int(elem_back.quarterLength * 24) # Decrement position
            elem_back.offset = pos_j / 24 # Then set offset
            out_back.insert(0, elem_back) # Then append

            idx += 1  # Increment index for both streams
            if idx >= max_tokens:
                out.extend(out_fwd + out_back)
                pos += len_split
                take = 'melody'
                continue
    
    return stream.Stream(out)

def main(args):
    mode = 'training' if args.train_model else 'inference'
    print(f"Loading ParkerGPT in {mode} mode...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        train_models(input_tokens, device, hp, model_save_dir)
 
    else: # inference mode
        # parse midi file
        melody_in: stream.Stream = parse_midi(args.input_path)
        print(f"Original melody: {melody_in.quarterLength} quarter notes long")

        # get simple triads (baseline for markov chain)
        print("\nSimple triads:")
        keysig, simple_triads = get_triads(melody_in)

        # generate the markov table ()
        print(f"\nGenerating markov table")
        markov_table = generate_markov_table() # TODO: this is too convoluted

        print(f"\nApplying {args.chord_gen_depth} markov generation cycles...")
        substituted_chords: stream.Stream = simple_triads
        for pos_i in range(args.chord_gen_depth):
            substituted_chords = substitute_chords(substituted_chords, markov_table)
        
        substituted_chords = add_tail(substituted_chords)
        
        model_save_dir = args.model_save_dir
        if model_save_dir is None:
            model_save_dir = './'
        elif model_save_dir[-1] != '/':
            model_save_dir += '/'

        print(f"\nLoading trained model")
        try:
            checkpoint_f = torch.load(model_save_dir + 'parkergpt_forward.pt',
                                      map_location=device)
            checkpoint_b = torch.load(model_save_dir + 'parkergpt_backward.pt',
                                      map_location=device)
        except:
            print("Cannot find [parkergpt_forward.pt] and [parkergpt_backward.pt] " + 
                  f"in directory '{model_save_dir}'")
            print("try specifying models' directory using '--model_save_dir [DIR]'")
            quit()
        
        model_f = BigramModel(
            device=device,
            block_size=checkpoint_f['block_size'],
            vocab_size=checkpoint_f['vocab_size'],
            n_embed=checkpoint_f['n_embed']
        )
        model_b = BigramModel(
            device=device,
            block_size=checkpoint_b['block_size'],
            vocab_size=checkpoint_b['vocab_size'],
            n_embed=checkpoint_b['n_embed']
        )
        model_f.load_state_dict(checkpoint_b['model_state_dict'])
        model_b.load_state_dict(checkpoint_b['model_state_dict'])
        print("Good.")

        print("\nGenerating melody...")
        cw = args.context_size
        sl = random.choice([7, 9, 15])
        print("context size:", cw)
        print("segment length:", sl)
        res_melody = splice_melody(model_f, model_b, device,
                                   melody_in,
                                   past_context=cw,
                                   future_context=(cw + 1) // 2,
                                   max_tokens=20,
                                   segment_length=sl,
                                   crossover_threshold=25)
        
        out_score = stream.Score()
        out_score.insert(0, res_melody)
        out_score.insert(0, substituted_chords.transpose(-12))

        out_score.write('midi', args.output_path)
        print(f"output saved to '{args.output_path}'")
        print(f"\nProgram terminating.")


def __main__():
    parser = argparse.ArgumentParser(description='Jazzify a monophonic melody with ParkerGPT')

    # training args
    parser.add_argument('--train_model', action='store_true', default=False,
                        help="Use this to run the program in 'training' mode")
    parser.add_argument('--model_config', type=str, metavar='CFG', default=None,
                        help='[YAML] config for training hyperparameters')
    parser.add_argument('--train_data_dir', type=str, metavar='DIR', default=None,
                        help='[Path] to directory with musicxml files')
    parser.add_argument('--model_save_dir', type=str, metavar='DIR', default='./',
                        help='[Path] to directory where trained models will be saved to/loaded from')

    # inference args
    parser.add_argument('--input_path', type=str, metavar='FP', default=None,
                        help="[File Path] to input '.mid' melody (for inference mode)")
    parser.add_argument('--chord_gen_depth', type=int, metavar='D',
                        help='[Integer] number of markov generation cycles to run', default=5)
    parser.add_argument('--context_size', type=int, metavar='S', default=3,
                        help='[Integer] size of the context window for inference')
    parser.add_argument('--output_path', type=str, metavar='FP', default='./parkergpt_output.mid',
                        help="[File Path] to an output '.mid' melody")

    args = parser.parse_args()
    return main(args)

if __name__ == '__main__':
    __main__()

