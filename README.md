# Music Generation Using LSTM

## Overview

This project generates music using a Long Short-Term Memory (LSTM) neural network. The model is trained on MIDI files, learns note patterns, and generates new musical sequences.

## Features

* Extracts notes from MIDI files
* Preprocesses music data for training
* Trains an LSTM-based deep learning model
* Generates new note sequences
* Creates a MIDI file containing AI-generated music

## Technologies Used

* Python
* TensorFlow / Keras
* Music21
* NumPy

## Project Workflow

1. Collect MIDI music files.
2. Extract musical notes from MIDI files.
3. Convert notes into training sequences.
4. Train an LSTM model on the extracted notes.
5. Generate new note sequences.
6. Save generated music as a MIDI file.

## Files

* `extract_notes.py` – Extracts notes from MIDI files.
* `train_model.py` – Trains the LSTM model.
* `generate_music.py` – Generates new music using the trained model.
* `requirements.txt` – Project dependencies.

## How to Run

Install dependencies:

pip install -r requirements.txt

Extract notes:

python extract_notes.py

Train the model:

python train_model.py

Generate music:

python generate_music.py

## Output

The project generates a new MIDI file containing AI-generated music based on learned note patterns.
