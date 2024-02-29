import torch
import sys
from .Model_ablation import *

import argparse
import numpy as np
from moviepy.editor import VideoFileClip
import os

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

def main():
    import pkg_resources

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



    videoPath = args.videoPath
    # Extract and save the audio
    audioPath = args.videoPath.replace(".mp4", ".mp3")
    extractAudio(args.videoPath, audioPath)
    print("DONE")
    outPath = args.outPath
    torch.set_grad_enabled(False)


    # Complete visual model code
    candidate_notes = processVideo(videoPath)
    print(candidate_notes)

    fs, audio = readAudio(audioPath)


    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
)



    x = torch.from_numpy(audio).to(device)


    notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False, candidateNotes=candidate_notes)

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)

def processVideo(path):
    root, _ = os.path.splitext(path)
    vision = PianoVision(root)
    candidate_notes = vision.main_loop()
    return candidate_notes

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
