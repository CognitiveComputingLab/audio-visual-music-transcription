import os
from mido import MidiFile, merge_tracks

def merge_midi_tracks_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            midi_file = MidiFile(input_file_path)

            merged_track = merge_tracks(midi_file.tracks)

            new_midi_file = MidiFile()
            new_midi_file.tracks.append(merged_track)

            new_midi_file.save(output_file_path)

# Example usage
input_folder = 'complete/MIDI_new'
output_folder = 'complete/MIDI_merged'
merge_midi_tracks_in_folder(input_folder, output_folder)
