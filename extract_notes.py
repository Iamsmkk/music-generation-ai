from music21 import converter, instrument, note, chord
import os
import pickle

notes = []

dataset_path = "dataset"

for file in os.listdir(dataset_path):

    if file.endswith(".mid"):

        print("Reading:", file)

        midi = converter.parse(os.path.join(dataset_path, file))

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

print("\nTotal Notes:", len(notes))

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("notes.pkl created successfully")