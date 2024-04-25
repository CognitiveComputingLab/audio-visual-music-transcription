import os
import mido
from mido import MidiFile, MidiTrack, Message

# Similar to before but separate tempo tracks
def txt_to_midi(input_file, output_file, velocity=64):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    note_offset = 0

    with open(input_file, 'r') as file:
        for line in file:
            note_data = line.split('\t')
            midi_note = int(note_data[2]) + note_offset

            note_on_message = Message("note_on", note=midi_note, velocity=velocity, time=int(float(note_data[0]) * 1000))
            note_off_message = Message("note_off", note=midi_note, velocity=velocity, time=int((float(note_data[1]) - float(note_data[0])) * 1000))

            new_track = MidiTrack()
            mid.tracks.append(new_track)
            
            new_track.append(note_on_message)
            new_track.append(note_off_message)

    mid.save(output_file)

def convert_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename.replace(".txt", ".mid"))
            txt_to_midi(input_file_path, output_file_path)

convert_folder("complete/text", "complete/MIDI_new")
