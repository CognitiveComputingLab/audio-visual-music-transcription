# audio-visual-music-transcription

Models
- Fuse "Skipping-the-Frame-Level" and "piano-vision"
- Audio-Visual Model V1 --> enhanced with standard computer vision techniques
- Audio-Visual Model V2 --> enhanced with deep keyboard segmentation
- Application --> V1 with the option to choose the exact keyboard region

Main changes: 
- For the audio model, implementations have been made in transkun/transcribe.py and transkun/Model_ablation.py
- For the visual model, see main.py, and the processors folder

Datasets
- pianoDetectData (keyboard segmentation training)
- OMAPS (evaluation)
- OMAPS2 (evaluation with velocity)

Evaluation
- MIR Evaluation
- MV2H (source code omitted)