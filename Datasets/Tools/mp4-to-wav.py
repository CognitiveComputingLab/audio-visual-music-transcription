from moviepy.editor import *
import os

in_dir = "complete/mp4"
out_dir = "complete/wav"

os.makedirs(out_dir, exist_ok=True)

for filename in os.listdir(in_dir):
    if filename.endswith(".mp4"):
        input_path = os.path.join(in_dir, filename)
        output_path = os.path.join(out_dir, filename.replace(".mp4", ".wav"))

        video = VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(output_path)
        video.close()

print("DONE")