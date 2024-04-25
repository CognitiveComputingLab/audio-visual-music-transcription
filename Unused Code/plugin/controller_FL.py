import interface
import model_MIDI_handler

def main():
    # The GUI funcions
    
    # If a file has been uploaded
        # Link to the selected video file
        video = interface.video_file()

        # Run the video through the audio-visual model to generate the MIDI
        midi = model_MIDI_handler.generate(video)

        # Update FL studios with the MIDI file
        interface.add_MIDI(midi)

if __name__ == "__main__":
    main()