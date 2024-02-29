import pandas as pd
import pickle
from pathlib import Path
import sys
import os
from Data import Note

df = pd.read_csv("../OMAPS2/OMAPS2.csv", header=0)
print(df)

# Read every note
def notes_read(filename):
    # Add the path correctly
    path = Path("../OMAPS2/complete/text") / filename
    # Note array
    notes = []
    with open(path, 'r') as file:
        for line in file:
            # The columns defined in read me
            start, end, pitch, velocity = line.split()
            notes.append(Note(
                start=float(start),
                end=float(end),
                pitch=int(pitch),
                velocity=int(velocity)
            ))
    return notes

fs = 44100
nChannel = 1

# Create a list of dictionaries
data = []
for _, row in df.iterrows():
    midi_filename = row["midi_filename"]
    txt_filename = midi_filename.split('/')[-1].replace(".mid", ".txt")
    notes = notes_read(txt_filename)

    nSamples = int(float(row["duration"]) * fs)
    
    # Add the the information required
    data.append({
        'split': row['split'],
        'midi_filename': row['midi_filename'],
        'audio_filename': row['audio_filename'],
        'duration': row['duration'],
        'notes': notes,
        'fs': fs,
        'nSamples': nSamples,
        'nChannel': nChannel
    })

# Turn into a df
df_data = pd.DataFrame(data)

# Save the train, test and valiation sections into separate pickle files
for split_name, group_df in df_data.groupby("split"):
    group_data = group_df.to_dict("records")

    # Add to the right path
    file_path = os.path.join("transkun", f'{split_name}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(group_data, file)

print("Pickled.")