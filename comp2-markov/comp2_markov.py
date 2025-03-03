import itertools
from numpy.random import choice
from random import random
from music21 import *

def parse_midi(filepath) -> stream.Stream:
    midi = converter.parse(filepath)
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
            # Convert chord to its highest note
            return_stream.append(note.Note(element.pitches[0]))
        elif isinstance(element, note.Rest):
            # Preserve rests
            return_stream.append(element)
    return return_stream

def get_triads(midi: stream.Stream) -> stream.Stream:
    keysig: key.Key = midi.analyze("key")
    scale = keysig.getScale()
    
    measures = midi.makeMeasures()

    out_stream: stream.Stream = stream.Stream()
    out_stream.append(keysig)

    for measure in measures:
        if len(measure.notes) == 0:
            out_stream.append(note.Rest(quarterLength=4.0))
            continue
        tonic: note.Note = measure.notes[0]
        # build chord using scale degrees from key signature
        chord_pitches = [tonic.pitch,
                         scale.nextPitch(tonic.pitch, 2),
                         scale.nextPitch(tonic.pitch, 4)]
        chord_pitches = [p.transpose(-12) for p in chord_pitches]
        out_stream.append(chord.Chord(chord_pitches, quarterLength=4.0))

    print([roman.romanNumeralFromChord(c, keysig).figure for c in
               out_stream.recurse().getElementsByClass('Chord')])

    return out_stream


def generate_markov_table() -> dict[tuple, dict[str, float]]:
    possible_chords = ["I", "i", "II", "ii", "III", "iii", "IV", "iv", "V", "v", "VI", "vi", "VII", "vii"]

    substitution_table = {}

    for state in itertools.product(possible_chords, repeat=3):
        substitution_distribution = {}
        chord_default = state[2]
        idx_0 = (((possible_chords.index(state[0]) // 2) + 3) * 2) % len(possible_chords)
        valid_degrees = [idx_0, idx_0 + 1]
        idx_1 = possible_chords.index(state[1])
        target_chord = possible_chords[(idx_1 + 6) % len(possible_chords)]
        # if 2nd chord's index is 3 degrees higher
        if idx_1 in valid_degrees:
            target_weight = 0.5
            remaining_weight = 1 - target_weight
            for chord in possible_chords:
                if chord == target_chord:
                    substitution_distribution[chord] = target_weight
                else:
                    substitution_distribution[chord] = remaining_weight / (len(possible_chords) - 1)
        else:
            default_prob = 0.825
            target_prob = 0.125 # chance of moving to the target chord anyway
            combined = chord_default == target_chord
            primary_prob = default_prob + (target_prob if combined else 0)
            secondary_prob = target_prob if not combined else 0
            other_prob = (1 - primary_prob - secondary_prob) / (len(possible_chords) - (1 if combined else 2))
            
            for chord in possible_chords:
                if chord == chord_default:
                    substitution_distribution[chord] = primary_prob
                elif chord == target_chord and not combined:
                    substitution_distribution[chord] = secondary_prob
                else:
                    substitution_distribution[chord] = other_prob
        substitution_table[state] = substitution_distribution

    print("Total number of states:", len(substitution_table))
    return substitution_table

def substitute_chords(chord_stream: stream.Stream, markov_table: dict[tuple, dict[str, float]]) -> stream.Stream:
    keysig: key.Key = chord_stream[0]
    scale = keysig.getScale()
    chord_numerals = [roman.romanNumeralFromChord(c, keysig).romanNumeralAlone for c in
                          chord_stream.recurse().getElementsByClass('Chord')]
    out_stream = stream.Stream()
    out_chords = []
    i = 0
    for c in chord_stream.recurse():
        if isinstance(c, key.Key):
            out_stream.append(c)
        elif isinstance(c, note.Rest):
            out_stream.append(c)
        elif isinstance(c, chord.Chord):
            state = tuple(chord_numerals[max(i-2,0):i+1])
            while len(state) < 3:
                state = tuple([chord_numerals[0]]) + state # prepend first chord
            # print(f"state: {state}")
            substitution_distribution = markov_table[state]
            # print(f"probabilities: {substitution_distribution}")
            new_chord_symbol = choice(list(substitution_distribution.keys()),
                                    p=list(substitution_distribution.values()))
            if random() < 0.5:
                new_chord_symbol += '2'
            elif random() < 0.5:
                new_chord_symbol += '7'
            elif random() < 0.2:
                new_chord_symbol += 'sus4'
            if '2' not in new_chord_symbol and 's' not in new_chord_symbol:
                if random() < 0.2:
                    new_chord_symbol += '6'
            out_chords.append(new_chord_symbol)
            new_chord = (chord.Chord(roman.RomanNumeral(new_chord_symbol, keysig),
                        quarterLength=4.0)).transpose(-12)
            out_stream.append(new_chord)
            # print(f"new chord: {out_stream[-1]}")
            i += 1
    print("\nnew chords:")
    print(out_chords)
    return out_stream
