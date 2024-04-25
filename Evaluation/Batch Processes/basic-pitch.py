import os
from basic_pitch.inference import predict_and_save

input_directory = "OMAPS2/complete/wav_converted"
output_directory = "OMAPS2/evaluation/basic-pitch"

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):

        input_audio_path = os.path.join(input_directory, filename)

        input_audio_path_list = [input_audio_path]

        # Convert wav to midi
        try:
            predict_and_save(
                input_audio_path_list,
                output_directory,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False
            )
        except Exception as e:
            print("ERROR")

print("DONE")

# import subprocess
# import os

# output_directory = "OMAPS/evaluation/basic-pitch"
# input_directory = 'OMAPS/complete/wav_converted'

# os.makedirs(output_directory, exist_ok=True)

# for i in range(1, 107):
#     padded_number = f"{i:03d}"
#     input_path = os.path.join(input_directory, f"{padded_number}.wav")
#     if os.path.exists(input_path):
#         command = ["basic-pitch", output_directory, input_path]
#         subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# print("DONE")


# basic-pitch "OMAPS/evaluation/basic-pitch" "OMAPS/complete/wav_converted/001.wav" "OMAPS/complete/wav_converted/002.wav" "OMAPS/complete/wav_converted/003.wav" likewise does not work