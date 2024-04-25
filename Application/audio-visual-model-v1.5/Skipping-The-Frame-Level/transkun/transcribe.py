import torch
import sys
from .Model_ablation import *

import argparse
import numpy as np
from moviepy.editor import VideoFileClip
import os

import time

import concurrent.futures
import pkg_resources

# Add the parent directory to the path
sys.path.append("../piano-vision")

from piano_vision.main import PianoVision

def extractAudio(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

def readAudio(path,  normalize= True):
    import pydub
    audio = pydub.AudioSegment.from_mp3(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y =  np.float32(y)/2**15
    return audio.frame_rate, y

def processVideo(path):
    root, _ = os.path.splitext(path)
    vision = PianoVision(root)
    candidate_notes = vision.main_loop()
    return candidate_notes

def main():
    defaultWeight =  (pkg_resources.resource_filename(__name__, "pretrained/0.1.pt"))

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("videoPath", help = "path to the input video file")
    argumentParser.add_argument("outPath", help = "path to the output MIDI file")
    argumentParser.add_argument("--weight", default = defaultWeight, help = "path to the pretrained weight")
    argumentParser.add_argument("--device", default = "cpu", nargs= "?", help = " The device used to perform the most computations (optional), DEFAULT: cpu")
    argumentParser.add_argument("--segmentHopSize", type=float, default = 10, help = " The segment hopsize for processing the entire audio file (s), DEFAULT: 10")
    argumentParser.add_argument("--segmentSize", type=float, default = 20, help = " The segment size for processing the entire audio file (s), DEFAULT: 20")

    args = argumentParser.parse_args()

    path = args.weight
    device = args.device
    checkpoint = torch.load(path, map_location = device)


    conf = TransKun.Config()
    conf.__dict__ = checkpoint['conf']

    model = TransKun(conf = conf).to(device)

    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()


    # Define the paths
    videoPath = args.videoPath
    audioPath = args.videoPath.replace(".mp4", ".mp3")

    # Extract and save the audio
    extractAudio(args.videoPath, audioPath)
    print("DONE")
    outPath = args.outPath
    torch.set_grad_enabled(False)

    fs, audio = readAudio(audioPath)
    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
        )


    x = torch.from_numpy(audio).to(device)


    # start = time.time()
    # candidate_notes = processVideo(videoPath)
    # print(candidate_notes)

    # # Complete the audio model code
    # notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False, candidateNotes=candidate_notes)
    # end = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            start = time.time()
            # Complete visual model code
            future = executor.submit(processVideo, videoPath)
            #candidate_notes = processVideo(videoPath)
            candidate_notes = future.result()
        except:
            candidate_notes = list(range(21, 109))
            candidate_notes = [candidate_notes]

        print(candidate_notes)
        # Complete the audio model code
        notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False, candidateNotes=candidate_notes)
        end = time.time()

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)

    video_path_base = Path(args.outPath).stem
    output_name = video_path_base + ".txt"

    output_directory = Path('./output')
    output_directory.mkdir(parents=True, exist_ok=True)

    output_txt_path = output_directory / output_name

    time_to_complete = end - start

    with output_txt_path.open('w') as file:
        file.write(str(candidate_notes) + '\n')
        file.write(str(time_to_complete) + '\n')

if __name__ == "__main__":
    main()


# import torch
# from .Model_ablation import *


# import argparse


# def readAudio(path,  normalize= True):
#     import pydub
#     audio = pydub.AudioSegment.from_mp3(path)
#     y = np.array(audio.get_array_of_samples())
#     y = y.reshape(-1, audio.channels)
#     if normalize:
#         y =  np.float32(y)/2**15
#     return audio.frame_rate, y


# def main():
#     import pkg_resources

#     defaultWeight =  (pkg_resources.resource_filename(__name__, "pretrained/0.1.pt"))

#     argumentParser = argparse.ArgumentParser()
#     argumentParser.add_argument("audioPath", help = "path to the input audio file")
#     argumentParser.add_argument("outPath", help = "path to the output MIDI file")
#     argumentParser.add_argument("--weight", default = defaultWeight, help = "path to the pretrained weight")
#     argumentParser.add_argument("--device", default = "cpu", nargs= "?", help = " The device used to perform the most computations (optional), DEFAULT: cpu")
#     argumentParser.add_argument("--segmentHopSize", type=float, default = 10, help = " The segment hopsize for processing the entire audio file (s), DEFAULT: 10")
#     argumentParser.add_argument("--segmentSize", type=float, default = 20, help = " The segment size for processing the entire audio file (s), DEFAULT: 20")

#     args = argumentParser.parse_args()

#     path = args.weight
#     device = args.device
#     checkpoint = torch.load(path, map_location = device)


#     conf = TransKun.Config()
#     conf.__dict__ = checkpoint['conf']

#     model = TransKun(conf = conf).to(device)

#     if not "best_state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["state_dict"], strict=False)
#     else:
#         model.load_state_dict(checkpoint["best_state_dict"], strict=False)

#     model.eval()


#     audioPath = args.audioPath
#     outPath = args.outPath
#     torch.set_grad_enabled(False)


#     fs, audio= readAudio(audioPath)


#     if(fs != model.fs):
#         import soxr
#         audio = soxr.resample(
#                 audio,          # 1D(mono) or 2D(frames, channels) array input
#                 fs,      # input samplerate
#                 model.fs# target samplerate
# )



#     x = torch.from_numpy(audio).to(device)


#     notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False)

#     outputMidi = writeMidi(notesEst)
#     outputMidi.write(outPath)


# if __name__ == "__main__":
#     main()
