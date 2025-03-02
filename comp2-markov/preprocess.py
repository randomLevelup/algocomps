from music21 import *
import os
from tqdm import tqdm

def load_score(filepath):
    return converter.parse(filepath)

def split_long_notes(pitch, duration_24ths):
    """split notes longer than a quarter note into smaller segments"""
    segments = []
    remaining = duration_24ths
    
    while remaining > 0:
        if remaining > 24:
            segments.append((pitch, 24))  # add a quarter note
            remaining -= 24
        else:
            segments.append((pitch, remaining))  # add the remaining duration
            remaining = 0
            
    return segments

def score_to_sequence(score, transpose_interval=0):
    """convert a music21 score to a sequence of (pitch, duration) tuples"""
    if isinstance(score, list):
        notes_and_rests = score
    else:
        melody_part = score.parts[0]
        melody_part = melody_part.transpose(transpose_interval)  # transpose for variation
        
        # filter out chord symbols and get only notes and rests
        notes_and_rests = [elem for elem in melody_part.flatten().notesAndRests 
                        if isinstance(elem, (note.Note, note.Rest))]
    
    processed = []
    
    for element in notes_and_rests:
        duration_24ths = int(element.duration.quarterLength * 24)
        
        if isinstance(element, note.Note):
            if hasattr(element, 'pitch'):
                segments = split_long_notes(element.pitch.midi, duration_24ths)
                processed.extend(segments)
        elif isinstance(element, note.Rest):
            segments = split_long_notes(-1, duration_24ths)
            processed.extend(segments)
            
    return processed

def sequence_to_score(sequence):
    """convert a sequence of (pitch, duration) tuples back to a music21 stream"""
    output_stream = stream.Stream()
    
    for midi_num, duration_24ths in sequence:
        duration = duration_24ths / 24.0  # convert 24ths to quarter notes
        
        if midi_num == -1:
            output_stream.append(note.Rest(quarterLength=duration))
        else:
            output_stream.append(note.Note(midi_num, quarterLength=duration))
    
    return output_stream

def preprocess_data(data_dir, variation_amt=1):
    processed_sequences = []
    scores = []

    progress_bar = tqdm(desc="Loading training files", total=len(os.listdir(data_dir)))

    # collect all scores
    for filename in os.listdir(data_dir):
        if filename.endswith('.xml') or filename.endswith('.musicxml'):
            filepath = os.path.join(data_dir, filename)
            score = load_score(filepath)
            scores.append(score)
            progress_bar.update(1)
    
    progress_bar.close()

    total_iterations = variation_amt * len(scores)
    progress_bar = tqdm(desc="Processing training files", total=total_iterations)
    
    # generate variations for each
    for i in range(variation_amt):
        for score in scores:
            sequence = score_to_sequence(score, i)
            processed_sequences.append(sequence)
            progress_bar.update(1)
    
    progress_bar.close()

    return processed_sequences

def make_tokens(sequence: list[tuple[int, int]]) -> list[int]:
    """convert (pitch, duration) tuples to integer tokens"""
    MAX_DURATION = 25  # allows durations 1-24 inclusive
    
    tokens = []
    for pitch, duration in sequence:
        # handle edge cases
        if pitch < -1: pitch = -1  # too-low pitches become rests
        elif pitch > 127: pitch = 127
        duration = min(24, max(1, duration))
        
        # convert to token: shift pitch by 2 to handle -1 for rests
        token = (pitch + 2) * MAX_DURATION + duration
        tokens.append(token)
    
    return tokens

def tokens_to_sequence(tokens: list[int]) -> list[tuple[int, int]]:
    """convert integer tokens back to (pitch, duration) tuples"""
    MAX_DURATION = 25
    
    sequence = []
    for token in tokens:
        pitch = (token // MAX_DURATION) - 2
        duration = token % MAX_DURATION
        sequence.append((pitch, duration))
    
    return sequence

