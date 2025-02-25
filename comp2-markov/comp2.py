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
