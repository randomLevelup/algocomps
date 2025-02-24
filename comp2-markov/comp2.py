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
        elif hasattr(element, 'duration'):
            # Preserve non-note elements that have duration (like rests)
            return_stream.append(element)
    return return_stream

def get_downbeat_triads(midi: stream.Stream) -> tuple[str, list[str]]:
    keysig = midi.analyze("key")
    triads = []
    
    # Split into measures
    measures = midi.makeMeasures()
    
    for measure in measures:
        # Get first 3 notes in measure
        notes = list(measure.notes)[:3]
        if len(notes) < 3:
            triads.append("")
        else:
            # Create chord from first 3 notes
            measure_chord = chord.Chord([n.pitch for n in notes])
            
            # Convert to roman numeral
            rn = roman.romanNumeralFromChord(measure_chord, keysig)
            triads.append(rn.figure)
    
    return keysig, triads

