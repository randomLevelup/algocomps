from music21 import *
import os
import random

def load_score(filepath):
    return converter.parse(filepath)

def split_long_notes(pitch, duration_24ths):
    """
    Split notes longer than a quarter note into multiple quarter-note-or-shorter segments.
    """
    segments = []
    remaining = duration_24ths
    
    while remaining > 0:
        # If remaining duration is more than a quarter note
        if remaining > 24:
            segments.append((pitch, 24))  # Add a quarter note
            remaining -= 24
        else:
            segments.append((pitch, remaining))  # Add the remaining duration
            remaining = 0
            
    return segments

def score_to_sequence(score, transpose_interval=0):
    processed = []
    melody_part = score.parts[0]

    # transpose randomly for variation
    melody_part = melody_part.transpose(transpose_interval)
    
    # Filter out chord symbols and get only notes and rests
    notes_and_rests = [elem for elem in melody_part.flatten().notesAndRests 
                       if isinstance(elem, (note.Note, note.Rest))]
    
    for element in notes_and_rests:
        duration_24ths = int(element.duration.quarterLength * 24)
        
        if isinstance(element, note.Note):
            if hasattr(element, 'pitch'):
                # Split long notes into quarter-note-or-shorter segments
                segments = split_long_notes(element.pitch.midi, duration_24ths)
                processed.extend(segments)
        elif isinstance(element, note.Rest):
            # Split long rests into quarter-note-or-shorter segments
            segments = split_long_notes(-1, duration_24ths)
            processed.extend(segments)
            
    return processed

def sequence_to_score(sequence):
    """Convert a sequence of tuples (MIDI number, duration in 24ths) back to a music21 stream"""
    output_stream = stream.Stream()
    
    for midi_num, duration_24ths in sequence:
        duration = duration_24ths / 24.0  # Convert 24th notes back to quarter notes
        
        if midi_num == -1:
            output_stream.append(note.Rest(quarterLength=duration))
        else:
            output_stream.append(note.Note(midi_num, quarterLength=duration))
    
    return output_stream

def preprocess_data(data_dir, variation_amt=1):
    processed_sequences = []
    scores = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.xml') or filename.endswith('.musicxml'):
            filepath = os.path.join(data_dir, filename)
            score = load_score(filepath)
            scores.append(score)

    for i in range(variation_amt): # generate transposed variations
        for score in scores:
            sequence = score_to_sequence(score, i)
            processed_sequences.append(sequence)

    return processed_sequences

def make_tokens(sequence: list[tuple[int, int]]) -> list[int]:
    """
    Convert a sequence of (pitch, duration) tuples to integer tokens.
    Uses a deterministic formula with a smaller duration range (1-24).
    """
    # Maximum duration is 24 (quarter note)
    MAX_DURATION = 25  # Use 25 to allow durations 1-24 inclusive
    
    tokens = []
    for pitch, duration in sequence:
        # Ensure pitch is within MIDI range (-1 to 127)
        if pitch < -1:
            pitch = -1
        elif pitch > 127:
            pitch = 127
            
        # Ensure duration is within range (should be 1-24 after splitting)
        duration = min(24, max(1, duration))
        
        # Convert to token: shift pitch by 2 to handle -1 for rests
        token = (pitch + 2) * MAX_DURATION + duration
        tokens.append(token)
    
    return tokens

def tokens_to_sequence(tokens: list[int]) -> list[tuple[int, int]]:
    """
    Convert integer tokens back to a sequence of (pitch, duration) tuples.
    """
    MAX_DURATION = 25  # Must match the value used in make_tokens
    
    sequence = []
    for token in tokens:
        pitch = (token // MAX_DURATION) - 2
        duration = token % MAX_DURATION
        sequence.append((pitch, duration))
    
    return sequence

