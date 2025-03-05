"""

comp2.py
by jupiter westbard

"""

import random, argparse, itertools
from numpy.random import choice
from music21 import *

def parse_midi(filepath) -> stream.Stream:
    # load midi file
    try:
        midi = converter.parse(filepath)
    except:
        print(f"Input path [{filepath}] is not a valid MIDI file.")
        quit();

    # if multipart score, return the first part
    if isinstance(midi, stream.Score):
        midi = midi.parts[0]
    
    # make monophonic (convert all elements to single notes)
    midi = midi.flatten()
    return_stream = stream.Stream()
    for element in midi:
        if isinstance(element, note.Note):
            return_stream.append(element)
        elif isinstance(element, chord.Chord):
            # Flatten multiple notes to the lowest one
            return_stream.append(note.Note(element.pitches[0], duration=element.duration))
        elif isinstance(element, note.Rest):
            # Preserve rests
            return_stream.append(element)
    return return_stream

def get_triads(input: stream.Stream) -> tuple[key.Key, stream.Stream]:
    # guess key based on input notes
    keysig: key.Key = input.analyze("key")
    scale = keysig.getScale()
    
    measures = input.makeMeasures()

    out_stream: stream.Stream = stream.Stream()
    out_stream.append(keysig)

    # make a whole-note chord for each measure in input
    for measure in measures:
        if len(measure.notes) == 0: # skip measures with no notes
            out_stream.append(note.Rest(quarterLength=4.0))
            continue

        # build chord using scale degrees from key signature
        tonic: note.Note = measure.notes[0] # use the first note of the measure as tonic
        chord_pitches = [tonic.pitch,
                         scale.nextPitch(tonic.pitch, 2),
                         scale.nextPitch(tonic.pitch, 4)]
        chord_pitches = [p.transpose(-12) for p in chord_pitches]
        out_stream.append(chord.Chord(chord_pitches, quarterLength=4.0))

    print([roman.romanNumeralFromChord(c, keysig).figure for c in
               out_stream.recurse().getElementsByClass('Chord')])

    return keysig, out_stream

def generate_markov_table() -> dict[tuple, dict[str, float]]:
    # possible chords are ALTERNATING major and minor voicings of all 7 triads
    possible_chords = ["I", "i", "II", "ii", "III", "iii", "IV", "iv", "V", "v", "VI", "vi", "VII", "vii"]

    substitution_table = {}

    for state in itertools.product(possible_chords, repeat=3): # 3 'state' chords for each mapping
        # state[0] and state[1] are effectively the 'context' for state[2]
        substitution_distribution = {}

        # default is just the unaltered chord
        default_chord = state[2]

        # target1_idx is the (index of the) MAJOR voicing of the chord 3 semitones up from the
        # first chord in the state (i.e. if state[0] is 'ii', then target1_idx is 'V').
        target1_idx = (((possible_chords.index(state[0]) // 2) + 3) * 2) % len(possible_chords)
        valid_degrees = [target1_idx, target1_idx + 1] # both major and minor voicings are valid
        state1_idx = possible_chords.index(state[1]) # state1_idx is at the second chord in current state
        
        # target chord for current state is idx_1 plus 6 degrees
        # (i.e. if state1_idx is 'V', then target_chord is 'I')
        target_chord = possible_chords[(state1_idx + 6) % len(possible_chords)]

        if state1_idx in valid_degrees: # if idx_1 is a valid progression from state[0]
            target_weight = 0.65 # high-ish chance of moving to the target chord
            remaining_weight = (1 - target_weight) / (len(possible_chords) - 1) # remaining probs for other chords
            for chord in possible_chords:
                if chord == target_chord:
                    # target chord gets the highest weight
                    substitution_distribution[chord] = target_weight
                else:
                    # all others get an equal weight totaling to the remaining weight
                    substitution_distribution[chord] = remaining_weight
        
        else: # if idx_1 is not a valid progression from state[0]
            if default_chord == target_chord:
                target_weight = 0
                default_weight = 1 # already at the target chord, no need to move
                remaining_weight = (1 - target_weight - default_weight) / \
                                   (len(possible_chords) - 1)
            else:
                target_weight = 0.125 # small chance of moving to the target chord anyway
                default_weight = 0.825 # large chance of sticking with the original chord
                # leaves 0.05 over for a chance of switching to another random chord
                remaining_weight = (1 - target_weight - default_weight) / \
                                   (len(possible_chords) - 2)
                
            for chord in possible_chords:
                if chord == default_chord:
                    substitution_distribution[chord] = default_weight
                elif chord == target_chord:
                    substitution_distribution[chord] = target_weight
                else:
                    substitution_distribution[chord] = remaining_weight

        substitution_table[state] = substitution_distribution

    print("Total number of states:", len(substitution_table))
    return substitution_table

def substitute_chords(chord_stream: stream.Stream, markov_table: dict[tuple, dict[str, float]]) -> stream.Stream:
    # convert chord stream to roman numeral representation
    keysig: key.Key = chord_stream[0]
    scale = keysig.getScale()
    chord_numerals = [roman.romanNumeralFromChord(c, keysig).romanNumeralAlone for c in
                          chord_stream.recurse().getElementsByClass('Chord')]
    
    out_stream = stream.Stream()
    out_chords = []
    i = 0
    for c in chord_stream.recurse(): # loop over elements in chord stream
        if isinstance(c, key.Key):
            out_stream.append(c)
        elif isinstance(c, note.Rest):
            out_stream.append(c)
        elif isinstance(c, chord.Chord):
            # state for current chord is the previous 'context' chords (up to 2)
            # state will be used as the key for the markov dict
            state = tuple(chord_numerals[max(0,i-2):i+1]) # load state
            while len(state) < 3:
                state = tuple([chord_numerals[0]]) + state # prepend first chord to fill length
            
            substitution_distribution = markov_table[state] # get probabilities for current chord/state
            new_chord_symbol = choice(list(substitution_distribution.keys()), # make a choice
                                      p=list(substitution_distribution.values())) # based on those probabilities
            
            # randomly add some cheeky color tones
            # these are not preserved across chain iterations
            if random.random() < 0.5:
                new_chord_symbol += '2'
            elif random.random() < 0.5:
                new_chord_symbol += '7'
            elif random.random() < 0.2:
                new_chord_symbol += 'sus4'
            if '2' not in new_chord_symbol and 'sus4' not in new_chord_symbol:
                if random.random() < 0.2:
                    new_chord_symbol += '6'
            
            # append chord to output stream
            out_chords.append(new_chord_symbol)
            new_chord = (chord.Chord(roman.RomanNumeral(new_chord_symbol, keysig),
                        quarterLength=4.0)).transpose(-12)
            out_stream.append(new_chord)
            i += 1
    print("\nnew chords:")
    print(out_chords)
    return out_stream

def main(args):
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

    return

def __main__():
    parser = argparse.ArgumentParser(description='Generate algorithmic music composition')
    parser.add_argument('input_path',help='Input MIDI file path')
    parser.add_argument('--chord_gen_depth', type=int, metavar='D',
                        help='[Integer] number of markov generation cycles to run', default=5)
    
    args = parser.parse_args()
    return main(args)

if __name__ == '__main__':
    __main__()

