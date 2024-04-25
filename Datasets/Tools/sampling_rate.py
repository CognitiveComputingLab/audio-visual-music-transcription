from pydub import AudioSegment
import os

input_folder = "test/wav"
output_folder = "test/wav_converted"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        audio = AudioSegment.from_wav(input_path)

        audio = audio.set_frame_rate(44100)

        audio.export(output_path, format="wav")

print("DONE")
