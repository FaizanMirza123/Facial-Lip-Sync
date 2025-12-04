# Facial Lip-Sync System

A real-time facial animation system that generates realistic lip-synced faces from static images, synchronized with speech from STT/TTS agents.

## Features

- **Real-time Lip Synchronization**: Syncs lip movements with audio in real-time
- **Custom Model Training**: Train your own lip-sync models on custom datasets
- **STT/TTS Integration**: Works with OpenAI Whisper, GPT-4, and custom TTS models
- **Realistic Face Rendering**: Generates natural-looking facial animations
- **Flexible Architecture**: Modular design for easy customization

## Project Structure

```
Facial-Lip-Sync/
├── models/                  # Neural network architectures
│   ├── lip_sync_model.py   # Main lip-sync model
│   ├── discriminator.py    # GAN discriminator for realism
│   └── face_encoder.py     # Face identity encoder
├── data/                    # Data processing and loading
│   ├── dataset.py          # Dataset classes
│   └── preprocessing.py    # Audio/video preprocessing
├── training/                # Training scripts
│   ├── train.py            # Main training loop
│   └── losses.py           # Custom loss functions
├── inference/               # Real-time inference
│   ├── realtime_sync.py    # Real-time lip-sync engine
│   └── renderer.py         # Face rendering pipeline
├── utils/                   # Utility functions
│   ├── audio_processing.py # Audio feature extraction
│   ├── face_detection.py   # Face detection & landmarks
│   └── video_utils.py      # Video I/O utilities
├── stt_tts/                 # STT/TTS integration
│   ├── whisper_stt.py      # OpenAI Whisper integration
│   └── tts_integration.py  # TTS model integration
├── configs/                 # Configuration files
│   └── config.yaml         # Default configuration
├── examples/                # Example scripts
│   ├── train_model.py      # Training example
│   └── run_inference.py    # Inference example
└── checkpoints/             # Model checkpoints (created during training)
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training Your Model

```python
python examples/train_model.py --config configs/config.yaml --data_dir /path/to/dataset
```

### 2. Running Inference

```python
python examples/run_inference.py --image face.jpg --audio speech.wav --output output.mp4
```

### 3. Real-time Demo

```python
from inference.realtime_sync import RealtimeLipSync

# Initialize
lip_sync = RealtimeLipSync(model_path='checkpoints/best_model.pth')

# Process image and audio stream
lip_sync.run(image_path='face.jpg', audio_stream=audio_stream)
```

## Model Architecture

The system uses a deep learning architecture inspired by Wav2Lip with custom enhancements:

1. **Audio Encoder**: Processes mel-spectrograms from speech
2. **Face Encoder**: Extracts identity features from the face image
3. **Lip-Sync Generator**: Generates lip movements synchronized with audio
4. **Face Renderer**: Renders the final lip-synced face

## Training Data

The model requires paired video-audio data for training:

- Video clips with clear facial views
- Corresponding audio tracks
- Recommended: 10-50 hours of diverse speakers

## Configuration

Edit `configs/config.yaml` to customize:

- Model hyperparameters
- Training settings
- Audio processing parameters
- STT/TTS settings

## License

MIT License

## Acknowledgments

- Inspired by Wav2Lip and related lip-sync research
- Uses OpenAI Whisper for STT
- Face detection powered by dlib and face-alignment
