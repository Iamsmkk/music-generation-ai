import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Create mapping
pitchnames = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(pitchnames)
}

sequence_length = 20

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]

    network_input.append(
        [note_to_int[n] for n in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

n_patterns = len(network_input)

print("Training Patterns:", n_patterns)

# Reshape
network_input = np.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

# Build Model
model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(
            network_input.shape[1],
            network_input.shape[2]
        )
    )
)

model.add(Dropout(0.2))

model.add(Dense(256, activation="relu"))

model.add(
    Dense(
        len(pitchnames),
        activation="softmax"
    )
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

print(model.summary())

# Train
model.fit(
    network_input,
    network_output,
    epochs=15,
    batch_size=32
)

# Save model
model.save("model.h5")

print("Model saved successfully!")