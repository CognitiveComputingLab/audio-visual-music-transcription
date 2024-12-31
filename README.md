# audio-visual-music-transcription

**Models**
- Fused "Skipping-the-Frame-Level" and "piano-vision"
- Audio-Visual Model V1 --> enhanced with standard computer vision techniques
- Audio-Visual Model V2 --> enhanced with deep keyboard segmentation
- Application --> V1 with the option to choose the exact keyboard region

**Main changes**:
- For the audio model, implementations have been made in transkun/transcribe.py and transkun/Model_ablation.py
- For the visual model, see main.py, and the processors folder

**Datasets**
- pianoDetectData (keyboard segmentation training)
- OMAPS (evaluation)
- OMAPS2 (evaluation with velocity)

**Evaluation**
- MIR Evaluation
- MV2H (source code omitted)

## Results
Further details and results can be found [here](https://github.com/hashimh4/audio-visual-piano-transcription).

## Contact
Please send me an email at [hashimhussain4242@gmail.com](mailto:hashimhussain4242@gmail.com) for further information. I am also open and eager to discuss any available work/collaboration opportunities as a recent graduate.
