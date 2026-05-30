import pickle
import numpy as np

from tensorflow.keras.models import load_model
from music21 import note, chord, stream

# Load extracted notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Create mappings
pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 20

network_input = []

for i in range(len(notes) - sequence_length):
    sequence = notes[i:i + sequence_length]
    network_input.append([note_to_int[n] for n in sequence])

# Load trained model
model = load_model("model.h5")

# Pick a random starting sequence
start = np.random.randint(0, len(network_input) - 1)

pattern = network_input[start]

prediction_output = []

print("Generating Notes...\n")

# Generate 100 notes
for note_index in range(100):

    prediction_input = np.reshape(
        pattern,
        (1, len(pattern), 1)
    )

    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(
        prediction_input,
        verbose=0
    )[0]

    # Random sampling based on probabilities
    index = np.random.choice(
        range(len(prediction)),
        p=prediction
    )

    result = int_to_note[index]

    prediction_output.append(result)

    print(result)

    pattern.append(index)
    pattern = pattern[1:]

# Convert generated notes into MIDI
offset = 0
output_notes = []

for pattern in prediction_output:

    if "." in pattern or pattern.isdigit():

        notes_in_chord = pattern.split(".")
        chord_notes = []

        for current_note in notes_in_chord:

            new_note = note.Note(int(current_note))
            new_note.offset = offset

            chord_notes.append(new_note)

        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)
        new_note.offset = offset

        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write(
    "midi",
    fp="generated_music.mid"
)

print("\ngenerated_music.mid created successfully!")